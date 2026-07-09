from dataclasses import dataclass

from configs.llm_config import Tiny1M3MConfig
import train_llm


@dataclass
class C(Tiny1M3MConfig):
    use_value_channel_gate: bool = True


if __name__ == "__main__":
    train_llm.main()
