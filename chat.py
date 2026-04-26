"""
Interactive Chat script for nanoGPT
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out-shakespeare-char' # 셰익스피어 모델 폴더
device = 'cuda' # GPU 사용
dtype = 'float16' # RTX 2070 최적화
max_new_tokens = 100 # 답변 길이
temperature = 0.8 # 창의성 조절
top_k = 200
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 모델 로드
print(f"Loading model from {out_dir}...")
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

# 인코더/디코더 설정
load_meta = False
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 채팅 시작 루프
print("\n" + "="*50)
print("  Shakespearean Chat Bot (nanoGPT)")
print("  종료하려면 'exit' 또는 'quit'을 입력하세요.")
print("="*50 + "\n")

# 기본 대화 문맥 (Few-shot Prompt)
system_prompt = "The following is a conversation between a User and a Shakespearean Actor.\n\n"

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    # 모델에게 전달할 전체 문맥 구성
    prompt = f"{system_prompt}User: {user_input}\nActor:"
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    
    print("Actor: ", end="", flush=True)
    
    # 답변 생성
    with torch.no_grad():
        with ctx:
            # 여기서는 실시간 스트리밍 대신 생성이 완료된 후 한 번에 출력 (간단하게)
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            full_response = decode(y[0].tolist())
            
            # 모델이 'Actor:' 뒤에 답변을 시작하므로 그 부분만 추출
            # (매우 단순한 추출 로직)
            response_only = full_response[len(prompt):].split('\n')[0]
            print(response_only)
    
    print("-" * 30)

print("\n대화를 종료합니다. 즐거웠습니다!")
