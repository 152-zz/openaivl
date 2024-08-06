import torch
import pickle
from model.llama_model import create_look_ahead_mask,bumblebee

def generate_text(model, tokenizer, input_text, device,max_length=50):
    # 将输入文本转换为 token ID
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
    print(input_ids)
    model.eval()
    model = model.to(device)

    # 生成循环
    for _ in range(max_length):
        # 获取模型的输出
        length = len(input_ids)
        mask = create_look_ahead_mask(length).to(device)
        outputs = model(input_ids,mask)

        # 获取最后一个 token 的预测
        next_token_embed = outputs[0, -1, :]

        # 选择概率最高的 token
        next_token_id = torch.argmax(next_token_embed).unsqueeze(0).unsqueeze(0)

        # 将新生成的 token 添加到输入中
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

        # 如果生成的 token 是结束符（例如，0），则停止生成
        if next_token_id.item() == 1:
            break

    # 将生成的 token ID 转换回文本
    generated_text = tokenizer.decode(input_ids[0])

    return generated_text


    # 示例使用
if __name__ == '__main__':
    #GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #推理示例
    input_text = "InternVL: Scaling up Vision Foundation"

    #加载模型
    embed_size = 512
    num_layers = 8
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = 500
    vocab_size = 14000

    model = bumblebee(
        vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    )
    weight = torch.load("output/zzyllm_final.pth")
    model.load_state_dict(weight)

    #加载分词器
    with open('output/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    #推理
    generated_text = generate_text(model, tokenizer, input_text,device)
    print(generated_text)