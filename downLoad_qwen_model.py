from modelscope import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

# ==================== 1. 下载模型 ====================
print("正在下载Qwen3-4B-Instruct-2507 模型...")

model_dir=snapshot_download(
    "Qwen/Qwen3-4B-Instruct-2507",
    cache_dir='./models',   #下载到本地的保存目录
    revision='master'
)
print(f"模型已下载到：{model_dir}")

# ==================== 2. 加载模型（4bit量化，节省显存）====================
# 配置4bit量化

quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,  #启动4bit量化
    bnb_4bit_compute_dtype=torch.float16,   #计算时用16位精度
    bnb_4bit_quant_type='nf4',  #NormalFloat4,是专门为神经网络设计的4bit量化方法
    bnb_4bit_use_double_quant=True, #双重量化，进一步压缩显存
)

#分词器
tokenizer=AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)

print("正在加载4bit量化模型...")
model=AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto"
)
print(f"模型加载完成! 设备：{model.device}")

# ==================== 3. 测试模型 ====================
print("="*50)
print("测试模型效果...")
print("="*50)

# 测试 prompt
prompt = """请根据以下新闻内容回答用户问题。

新闻内容：
沧州雄狮在中超第23轮比赛中客场2-0战胜河南队。上半场第17分钟，河南队球员牛梓屹自摆乌龙。第91分钟，刘鑫瑜前场抢断后远射锁定胜局。

用户问题：沧州雄狮 vs 河南 比赛结果如何？

请用简洁的语言回答。"""

messages=[
    {"role":"system","content":"你是一个专业的新闻问答助手，基于提供的新闻内容回答用户问题."},
    {"role":"user","content":prompt}
]
text=tokenizer.apply_chat_template(
    messages,
    tokenize=False,     #返回字符串（而不是数字ID列表）
    add_generation_prompt=True  #末尾加上 assistant 标记
)

inputs=tokenizer([text],return_tensors="pt").to(model.device)

# 生成回答
with torch.no_grad():
    outputs=model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
response=tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
print(f"\n模型回答：\n{response}")
with open('./qwen3_model_path.txt', 'w', encoding='utf-8') as f:
    f.write(model_dir)
print(f"\n 模型路径已保存到 ./qwen3_model_path.txt")