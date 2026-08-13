import os
import torch
from model import GPTConfig, GPT
import tiktoken

# -----------------------------------------------------------------------------
out_dir = 'out-agent-ft' # 학습된 모델 폴더
device = 'cuda' # or 'cpu'
dtype = 'bfloat16' # RTX 4060 Laptop (Ada, CC 8.9)에 최적화
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

print("-" * 50)
print("AI Agent Expert 모델과 대화를 시작합니다! (종료하려면 'exit' 입력)")
print("-" * 50)

while True:
    prompt = input("\nUser: ")
    if prompt.lower() == 'exit':
        break
    
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    
    print("\nAI Expert: ", end="")
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=200)
            # 입력값(prompt) 이후의 답변만 출력
            full_text = decode(y[0].tolist())
            response = full_text[len(prompt):].strip()
            print(response)
