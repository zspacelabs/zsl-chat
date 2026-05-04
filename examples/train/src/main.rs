use std::{
    cmp::max,
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
    },
};

use anyhow::bail;
use bunsen_ng::{
    modules::reflection::XmlModuleTree,
    training::optimizers::{
        GroupOptimizerAdaptor2,
        OptimizerGroup,
    },
};
use burn::{
    data::dataloader::DataLoader,
    grad_clipping::GradientClipping,
    lr_scheduler::{
        composed::{
            ComposedLrSchedulerConfig,
            SchedulerReduction,
        },
        cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::{
        Module,
        ParamId,
    },
    nn::loss::CrossEntropyLossConfig,
    optim::{
        AdamWConfig,
        MuonConfig,
    },
    prelude::Backend,
    record::CompactRecorder,
    tensor::{
        AsIndex,
        Slice,
        Tensor,
        backend::AutodiffBackend,
        bf16,
        s,
    },
    train::{
        ClassificationOutput,
        InferenceStep,
        Learner,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
        metric::{
            LearningRateMetric,
            LossMetric,
        },
    },
};
use clap::Parser;
use hashbrown::HashSet;
use rand::{
    SeedableRng,
    rngs::StdRng,
};
use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    disk_cache::WordchipperDiskCache,
};
use wordchipper_cli_util::logging::LogArgs;
use zsl_chat::gpt::gpt_model::{
    GPT,
    GPTConfig,
};
use zsl_chat_data::{
    dataloader::ChatDataLoader,
    tokens::{
        DenseTokenBlocksOptions,
        TokenBatchIteratorOptions,
    },
};
use zsl_data_cache::dataset::DatasetCacheConfig;

#[derive(Debug, Clone, clap::Args)]
pub struct TokenBatchOptionsArgs {
    /// The number of sequences to load per batch.
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// The maximum number of tokens in a sequence.
    #[arg(long, default_value_t = 2048)]
    pub batch_seq_len: usize,

    /// The minimum number of sequences to keep in the buffer
    /// before loading more sequences.
    #[arg(long, default_value_t = 1024)]
    pub min_buffer: usize,
}

impl TokenBatchOptionsArgs {
    pub fn options(&self) -> TokenBatchIteratorOptions {
        TokenBatchIteratorOptions {
            batch_size: self.batch_size,
            batch_seq_len: self.batch_seq_len,
            min_buffer: self.min_buffer,
        }
    }
}

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The embedding dimension size.
    #[clap(long, default_value = "768")]
    pub embedding_dim: usize,

    /// The pretrained vocabulary.
    #[clap(long, default_value = "openai:p50k_edit")]
    pub pretrained_vocab: String,

    /// Beginning-of-Sequence token.
    #[arg(long, default_value = "<|bos|>")]
    pub bos_token: String,

    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to the dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Learning rate
    #[arg(long, default_value_t = 5e-5)]
    pub learning_rate: f64,

    /*
    /// Warm-up epochs.
    #[arg(long, default_value_t = 5)]
    pub warmup_epochs: usize,
     */
    /// Enable cautious weight decay.
    #[arg(long, default_value = "false")]
    pub cautious_weight_decay: bool,

    /// Optimizer Weight decay.
    #[arg(long, default_value_t = 5e-2)]
    pub weight_decay: f32,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "100")]
    pub num_epochs: usize,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 4)]
    pub batch_size: usize,

    /// The training seq len.
    #[clap(long, default_value = "2048")]
    pub seq_len: usize,

    /// Grads accumulation size for processing
    #[arg(short, long, default_value_t = 1)]
    pub grads_accumulation: usize,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/zsl-chat")]
    pub artifact_dir: String,
}

fn ensure_artifact_dir(artifact_dir: &str) -> anyhow::Result<()> {
    let _ignored = std::fs::remove_dir_all(artifact_dir);
    std::fs::create_dir_all(artifact_dir)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // args.logging.setup_logging(3).unwrap();

    run::<burn::backend::Autodiff<burn::backend::cuda::Cuda<bf16>>>(&args)
}

fn run<B: AutodiffBackend>(args: &Args) -> anyhow::Result<()> {
    type T = u32;

    println!("{:#?}", args);

    // Remove existing artifacts before to get an accurate learner summary
    let artifact_dir: &str = args.artifact_dir.as_ref();
    ensure_artifact_dir(artifact_dir)?;

    let device: B::Device = Default::default();

    let data_cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir.clone());
    log::info!("DATASET CACHE: {:#?}", data_cache_config);
    let mut data_cache = data_cache_config.clone().init()?;

    let mut disk_cache = WordchipperDiskCache::default();

    let shards: Vec<usize> = {
        let max_shard = data_cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for slice in &args.shards {
            for idx in slice.into_iter() {
                let shard = idx.expect_elem_index(max_shard);
                collected.insert(shard);
            }
        }
        let mut shards: Vec<usize> = collected.into_iter().collect();
        shards.sort();
        shards
    };

    log::info!("Loading Shards: {shards:?}");
    let shard_paths = data_cache.load_shards(&shards)?;

    let validation_ratio = 0.10;
    let num_validation_shards: usize = max(
        ((shard_paths.len() as f64) * validation_ratio).ceil() as usize,
        1,
    );
    let num_training_shards = shard_paths.len() - num_validation_shards;

    let training_paths: Vec<PathBuf> = shard_paths[..num_training_shards].to_vec();
    let validation_paths: Vec<PathBuf> = shard_paths[num_training_shards..].to_vec();

    let mut vocab: UnifiedTokenVocab<T> =
        wordchipper::load_vocab(&args.pretrained_vocab, &mut disk_cache)?
            .vocab()
            .to_token_type()?;

    let max_token = vocab.max_token().unwrap();

    // This is a stupid hack.
    let mut vocab_size = vocab.len();
    let bos_token: T = {
        let specials = vocab.special_vocab_mut();
        if let Some(tok) = specials.lookup_token(args.bos_token.as_bytes()) {
            tok
        } else {
            let tok = max_token + 1;
            specials.add_str_word(&args.bos_token, tok);
            vocab_size += 1;
            tok
        }
    };
    let vocab = Arc::new(vocab);

    let tok = wordchipper::TokenizerOptions::default()
        .with_accelerated_lexers(true)
        .with_parallel(true)
        .build(vocab);

    let gpt_config = GPTConfig::new()
        //.with_n_layer(8)
        .with_vocab_size(vocab_size);

    let gpt: GPT<B> = gpt_config.init::<B>(&device);

    let host = GptHost { gpt };

    let dl_config = DenseTokenBlocksOptions {
        batch_seq_len: args.seq_len,
        batch_size: args.batch_size,
        bos: vec![bos_token],
        ..Default::default()
    };

    let training_data_loader: ChatDataLoader<B> = ChatDataLoader::new(
        training_paths,
        Some(Arc::new(Mutex::new(StdRng::seed_from_u64(0)))),
        &device,
        tok.clone(),
        dl_config.clone(),
    );

    let validation_data_loader: ChatDataLoader<B::InnerBackend> =
        ChatDataLoader::new(validation_paths, None, &device, tok.clone(), dl_config);

    // TODO: This is ... a hack.
    let iters_per_epoch = training_data_loader.num_items() * 500;

    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .linear(LinearLrSchedulerConfig::new(1e-10, 1.0, 300))
        .cosine(CosineAnnealingLrSchedulerConfig::new(
            args.learning_rate,
            iters_per_epoch * args.num_epochs,
        ))
        .with_reduction(SchedulerReduction::Prod)
        .init()
        .expect("Failed to initialize learning rate scheduler");

    let training = SupervisedTraining::new(
        artifact_dir,
        Arc::new(training_data_loader),
        Arc::new(validation_data_loader),
    )
    .grads_accumulation(args.grads_accumulation)
    .num_epochs(args.num_epochs)
    .metrics((LossMetric::new(), LearningRateMetric::new()))
    .with_file_checkpointer(CompactRecorder::new())
    .summary();

    let mut mtree = XmlModuleTree::build(&host);

    let all_params: HashSet<ParamId> = mtree.param_ids()?.into_iter().collect();
    // ==>
    //   let all_params: HashSet<ParamId> = mtree
    //       .query()
    //       .params()
    //       .to_param_ids()?
    //       .collect();
    //
    // ==>
    //   let muon_params: HashSet<ParamId> = mtree
    //       .query()
    //       .select("descendant-or-self::Param")
    //       .to_param_ids()?
    //       .collect();

    let muon_params: HashSet<ParamId> = mtree
        .select_params("GptHost/GPT/Vec[@name='h']")
        .filter("@rank=2")
        .to_param_ids()?
        .into_iter()
        .collect();
    // ==>
    //   let muon_params: HashSet<ParamId> = mtree
    //       .query()
    //       .select("GptHost/GPT/Vec[@name='h']")
    //       .params()
    //       .filter("@rank=2")
    //       .to_param_ids()?
    //       .collect();
    //
    // ==>
    //   let muon_params: HashSet<ParamId> = mtree
    //       .query()
    //       .select("GptHost/GPT/Vec[@name='h']/descendant-or-self::Param[@rank=2]"
    // )       .to_param_ids()?
    //       .collect();

    let adamw_params: HashSet<ParamId> = all_params
        .difference(&muon_params)
        .cloned()
        .collect::<HashSet<_>>();

    if muon_params.is_empty() {
        bail!("Muon parameters not found");
    }
    if adamw_params.is_empty() {
        bail!("AdamW parameters not found");
    }

    let optimizer = GroupOptimizerAdaptor2::new(
        vec![OptimizerGroup::from_adaptor(
            adamw_params,
            &AdamWConfig::new()
                .with_cautious_weight_decay(args.cautious_weight_decay)
                .with_weight_decay(args.weight_decay)
                .init::<B, GptHost<B>>(),
        )],
        vec![
            OptimizerGroup::from_adaptor(muon_params, &MuonConfig::new().init::<B, GptHost<B>>())
                .with_fixed_lr(args.learning_rate * 0.5),
        ],
    )
    .unwrap()
    .with_grad_clipping(GradientClipping::Value(2.0));

    let result = training.launch(Learner::new(host, optimizer, lr_scheduler));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}

#[derive(Module, Debug)]
pub struct GptHost<B: Backend> {
    pub gpt: GPT<B>,
}

impl<B: Backend> GptHost<B> {
    fn loss_step(
        &self,
        input: Tensor<B, 2, burn::prelude::Int>,
    ) -> ClassificationOutput<B> {
        let inputs: Tensor<B, 2, burn::prelude::Int> = input.clone().slice(s![.., ..-1]);
        let targets: Tensor<B, 2, burn::prelude::Int> = input.slice(s![.., 1..]);

        let mut kv_cache = None;

        // Logits.
        let outputs: Tensor<B, 3> = self.gpt.forward(inputs, &mut kv_cache);

        let output_flatten: Tensor<B, 2> = outputs.flatten(0, 1);
        let targets_flatten: Tensor<B, 1, burn::prelude::Int> = targets.flatten(0, 1);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_flatten.device())
            .forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for GptHost<B> {
    type Input = Tensor<B, 2, burn::prelude::Int>;
    type Output = ClassificationOutput<B>;

    fn step(
        &self,
        input: Self::Input,
    ) -> TrainOutput<Self::Output> {
        let classification_output = self.loss_step(input);
        let grads = classification_output.loss.backward();

        TrainOutput::new(self, grads, classification_output)
    }
}

impl<B: Backend> InferenceStep for GptHost<B> {
    type Input = Tensor<B, 2, burn::prelude::Int>;
    type Output = ClassificationOutput<B>;

    fn step(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        self.loss_step(input)
    }
}
