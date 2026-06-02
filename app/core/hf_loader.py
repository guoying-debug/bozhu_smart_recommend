import os

from app.core.config import BERT_MODEL_ID, HF_CACHE_DIR, LOCAL_BERT_MODEL_DIR
from transformers import BertModel, BertTokenizer


def _local_model_ready(model_dir: str) -> bool:
    required_files = ("config.json", "tokenizer_config.json")
    return os.path.isdir(model_dir) and all(
        os.path.exists(os.path.join(model_dir, name)) for name in required_files
    )


def load_bert_bundle():
    """
    优先从本地模型目录加载；本地没有时再通过已配置好的国内镜像下载，
    并将结果固化到本地方便后续容器离线启动。
    """
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_BERT_MODEL_DIR), exist_ok=True)

    load_plan = []
    if _local_model_ready(LOCAL_BERT_MODEL_DIR):
        load_plan.append((LOCAL_BERT_MODEL_DIR, True, "本地模型目录"))
    load_plan.append((BERT_MODEL_ID, False, "国内镜像缓存"))

    last_error = None
    for source, local_only, label in load_plan:
        try:
            tokenizer = BertTokenizer.from_pretrained(
                source,
                cache_dir=HF_CACHE_DIR,
                local_files_only=local_only,
            )
            model = BertModel.from_pretrained(
                source,
                cache_dir=HF_CACHE_DIR,
                local_files_only=local_only,
            )

            if not local_only:
                tokenizer.save_pretrained(LOCAL_BERT_MODEL_DIR)
                model.save_pretrained(LOCAL_BERT_MODEL_DIR)

            return tokenizer, model, label
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"无法加载 BERT 模型。已尝试本地目录和国内镜像缓存，最后错误: {last_error}"
    )
