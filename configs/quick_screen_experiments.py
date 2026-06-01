from dataclasses import dataclass

from configs.llm_config import Screen10M1MConfig, Screen10M20MConfig, Screen10M5MConfig


@dataclass
class Screen10M1MSwiGLUConfig(Screen10M1MConfig):
    ffn_variant: str = "swiglu"


@dataclass
class Screen10M1MZeroInitConfig(Screen10M1MConfig):
    zero_init_output_projections: bool = True


@dataclass
class Screen10M1MSoftcap30Config(Screen10M1MConfig):
    logit_softcap: float = 30.0


@dataclass
class Screen10M1MDropout05Config(Screen10M1MConfig):
    dropout: float = 0.05


@dataclass
class Screen10M1MEmbedResidual01Config(Screen10M1MConfig):
    embedding_residual_scale_init: float = 0.1


@dataclass
class Screen10M1MResidualScale01Config(Screen10M1MConfig):
    residual_scale_init: float = 0.1


@dataclass
class Screen10M5MSwiGLUConfig(Screen10M5MConfig):
    ffn_variant: str = "swiglu"


@dataclass
class Screen10M5MZeroInitConfig(Screen10M5MConfig):
    zero_init_output_projections: bool = True


@dataclass
class Screen10M5MSoftcap20Config(Screen10M5MConfig):
    logit_softcap: float = 20.0


@dataclass
class Screen10M5MSoftcap10Config(Screen10M5MConfig):
    logit_softcap: float = 10.0


@dataclass
class Screen10M5MSoftcap15Config(Screen10M5MConfig):
    logit_softcap: float = 15.0


@dataclass
class Screen10M5MSoftcap30Config(Screen10M5MConfig):
    logit_softcap: float = 30.0


@dataclass
class Screen10M5MSoftcap50Config(Screen10M5MConfig):
    logit_softcap: float = 50.0


@dataclass
class Screen10M5MDropout02Config(Screen10M5MConfig):
    dropout: float = 0.02


@dataclass
class Screen10M5MDropout10Config(Screen10M5MConfig):
    dropout: float = 0.10


@dataclass
class Screen10M5MEmbedResidual01Config(Screen10M5MConfig):
    embedding_residual_scale_init: float = 0.1


@dataclass
class Screen10M5MResidualScale05Config(Screen10M5MConfig):
    residual_scale_init: float = 0.5


@dataclass
class Screen10M5MLayerScaleConfig(Screen10M5MConfig):
    residual_scale_init: float = 0.1


@dataclass
class Screen10M5MBatch4Config(Screen10M5MConfig):
    batch_size: int = 4


@dataclass
class Screen10M5MNKV1Config(Screen10M5MConfig):
    n_kv_heads: int = 1


@dataclass
class Screen10M5MNKV4Config(Screen10M5MConfig):
    n_kv_heads: int = 4


@dataclass
class Screen10M20MSwiGLUConfig(Screen10M20MConfig):
    ffn_variant: str = "swiglu"


@dataclass
class Screen10M20MZeroInitConfig(Screen10M20MConfig):
    zero_init_output_projections: bool = True


@dataclass
class Screen10M20MSoftcap30Config(Screen10M20MConfig):
    logit_softcap: float = 30.0


@dataclass
class Screen10M20MDropout05Config(Screen10M20MConfig):
    dropout: float = 0.05


@dataclass
class Screen10M20MEmbedResidual01Config(Screen10M20MConfig):
    embedding_residual_scale_init: float = 0.1


@dataclass
class Screen10M20MResidualScale01Config(Screen10M20MConfig):
    residual_scale_init: float = 0.1


@dataclass
class Screen10M20MLayerScaleConfig(Screen10M20MConfig):
    residual_scale_init: float = 0.1


@dataclass
class Screen10M20MSoftcap20Config(Screen10M20MConfig):
    logit_softcap: float = 20.0


@dataclass
class Screen10M20MDropout10Config(Screen10M20MConfig):
    dropout: float = 0.1


@dataclass
class Screen10M20MBatch4Config(Screen10M20MConfig):
    batch_size: int = 4
