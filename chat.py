import os
import openai
import pdb
import json
from tools.vector import retrive_top_k
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
openai.api_key = OPENAI_API_KEY

# messages=[
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "Who won the world series in 2020?"},
#       {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#       {"role": "user", "content": "Where was it played?"}
#   ]


def chat(text):
    # 填补上下文
    messages = [
        {"role": "system", "content": "你是一个医生"},
    ]

    knowledges = retrive_top_k(text)
    m_knowledges = " ".join(
        [str(i+1) + "." + k for i, k in enumerate(knowledges)])
    m_knowledges = "用户的提问是:"+text+" 你有以下知识可以作为补充参考,需要根据用户提问选择有用的知识:" + m_knowledges
    m_knowledges = m_knowledges + " 请根据专业知识和常识判断用户可能患的病,给出具体病情分析和建议"

    messages.append({"role": "user", "content": m_knowledges})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return_text = completion.choices[0].message["content"]
    return_text = return_text.strip()
    print(return_text)
    print(knowledges)
    return [return_text, knowledges]
