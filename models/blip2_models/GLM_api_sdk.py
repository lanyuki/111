# 加了保存本次调用api实现假新闻验证的验证结果
# import json
# from zhipuai import ZhipuAI
# from tqdm import tqdm  # 导入 tqdm
#
# # 初始化ZhipuAI客户端
# client = ZhipuAI(api_key="37f403ad7e1e00048a738cc98742eae5.2XvzU7T35ArxyUbK")  # 填写您自己的APIKey
#
# def evaluate_news(news_list):
#     results = []
#     true_count = 0
#     false_count = 0
#     correct_count = 0
#     incorrect_count = 0
#
#     # 使用 tqdm 包装 news_list，显示进度条
#     for item in tqdm(news_list, desc="Evaluating News", unit="news"):
#         news = item['txt']
#         label = item['label']
#
#         try:
#             # 单个新闻的处理
#             response = client.chat.completions.create(
#                 model="glm-4-plus",  # 填写需要调用的模型编码
#                 messages=[
#                     {"role": "system",
#                      "content": "你是一个网络的新闻打假博主，你会依据你原有的知识和最近的一些新闻，判断一个新闻是真还是假的。此外，输出时请严格按照先说出真假，然后列出理由的形式输出，重要的是你的适应性很强，不需要知道新闻的全部，甚至只需要一段描述性的话语，你也能判断它的真假。如果没有完整的新闻，你就直接按照逻辑推理和一些情感的分析去判断。但是你不用复述以上段句话，你只需要知道这个身份并且严格遵守，发展你的技能吧"},
#                     {"role": "user", "content": news}
#                 ],
#             )
#             result = response.choices[0].message['content'].strip().lower()  # 获取模型输出并处理
#             results.append((news, result))
#
#             # 统计真实和虚假新闻
#             if label:
#                 true_count += 1  # 实际为真实
#                 if result == "真实":  # 假设模型输出"真实"表示正确判断
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1
#             else:
#                 false_count += 1  # 实际为虚假
#                 if result == "虚假":  # 假设模型输出"虚假"表示正确判断
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1
#         except Exception as e:
#             # 捕获异常，确保程序不中断
#             results.append((news, f"Error processing news: {str(e)}"))
#
#     return results, true_count, false_count, correct_count, incorrect_count
#
# def read_news_from_file(file_path):
#     news_list = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             news_list.append(json.loads(line))  # 逐行读取 JSONL 数据并解析
#     return news_list
#
# def save_results_to_file(results, output_path):
#     with open(output_path, 'w', encoding='utf-8') as file:
#         for news, result in results:
#             file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')  # 将每条新闻和结果写入文件
#
# def main():
#     # 指定文件路径
#     file_path = "C:/Users/lk/Desktop/GLM4/data/politifact_test.jsonl"  # 请将此处替换为你的文件路径
#     output_path = "C:/Users/lk/Desktop/GLM4/data/evaluation_results.jsonl"  # 输出结果文件路径
#
#     # 从文件读取新闻数据
#     news_dataset = read_news_from_file(file_path)
#
#     # 对新闻数据集进行批量测评
#     results, true_count, false_count, correct_count, incorrect_count = evaluate_news(news_dataset)
#
#     # 输出结果
#     for idx, (news, result) in enumerate(results):
#         print(f"News {idx + 1}: {result}")
#
#     # 输出统计结果
#     print(f"\n统计结果：")
#     print(f"真实新闻数量: {true_count}")
#     print(f"虚假新闻数量: {false_count}")
#     print(f"正确判断数量: {correct_count}")
#     print(f"错误判断数量: {incorrect_count}")
#
#     # 计算并输出准确率，防止除以零
#     total_judgments = correct_count + incorrect_count
#     if total_judgments > 0:
#         accuracy = (correct_count / total_judgments) * 100
#         print(f"准确率: {accuracy:.2f}%")
#     else:
#         print("没有进行任何判断，无法计算准确率。")
#
#     # 保存验证结果
#     save_results_to_file(results, output_path)
#     print(f"验证结果已保存到: {output_path}")
#
# if __name__ == "__main__":
#     main()
#
#
# 每十次保存
# import json
# from zhipuai import ZhipuAI
# from tqdm import tqdm  # 导入 tqdm
#
# # 初始化ZhipuAI客户端
# client = ZhipuAI(api_key="37f403ad7e1e00048a738cc98742eae5.2XvzU7T35ArxyUbK")  # 填写您自己的APIKey
#
#
# def evaluate_news(news_list, output_path, save_interval=10):
#     results = []
#     true_count = 0
#     false_count = 0
#     correct_count = 0
#     incorrect_count = 0
#
#     # 使用 tqdm 包装 news_list，显示进度条
#     for idx, item in enumerate(tqdm(news_list, desc="Evaluating News", unit="news")):
#         news = item['txt']
#         label = item['label']
#
#         try:
#             # 单个新闻的处理
#             response = client.chat.completions.create(
#                 model="glm-4-plus",  # 填写需要调用的模型编码
#                 messages=[
#                     {"role": "system",
#                      "content": "你是一个网络的新闻打假博主，你会依据你原有的知识和最近的一些新闻，判断一个新闻是真还是假的。此外，输出时请严格按照先说出真假，然后列出理由的形式输出，重要的是你的适应性很强，不需要知道新闻的全部，甚至只需要一段描述性的话语，你也能判断它的真假。如果没有完整的新闻，你就直接按照逻辑推理和一些情感的分析去判断。但是你不用复述以上段句话，你只需要知道这个身份并且严格遵守，发展你的技能吧"},
#                     {"role": "user", "content": news}
#                 ],
#             )
#             result = response.choices[0].message['content'].strip().lower()  # 获取模型输出并处理
#             results.append((news, result))
#
#             # 统计真实和虚假新闻
#             if label:
#                 true_count += 1  # 实际为真实
#                 if result == "真实":  # 假设模型输出"真实"表示正确判断
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1
#             else:
#                 false_count += 1  # 实际为虚假
#                 if result == "虚假":  # 假设模型输出"虚假"表示正确判断
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1
#         except Exception as e:
#             # 捕获异常，确保程序不中断
#             results.append((news, f"Error processing news: {str(e)}"))
#
#         # 每隔 `save_interval` 条新闻保存一次结果
#         if (idx + 1) % save_interval == 0:
#             save_results_to_file(results, output_path)
#             print(f"已保存前 {idx + 1} 条新闻的评估结果到: {output_path}")
#
#     return results, true_count, false_count, correct_count, incorrect_count
#
#
# def read_news_from_file(file_path):
#     news_list = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             news_list.append(json.loads(line))  # 逐行读取 JSONL 数据并解析
#     return news_list
#
#
# def save_results_to_file(results, output_path):
#     with open(output_path, 'w', encoding='utf-8') as file:
#         for news, result in results:
#             file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')  # 将每条新闻和结果写入文件
#
#
# def main():
#     # 指定文件路径
#     file_path = "C:/Users/lk/Desktop/GLM4/data/politifact_test.jsonl"  # 请将此处替换为你的文件路径
#     output_path = "C:/Users/lk/Desktop/GLM4/data/evaluation_results.jsonl"  # 输出结果文件路径
#
#     # 从文件读取新闻数据
#     news_dataset = read_news_from_file(file_path)
#
#     # 对新闻数据集进行批量测评，每 10 条保存一次结果
#     results, true_count, false_count, correct_count, incorrect_count = evaluate_news(news_dataset, output_path,
#                                                                                      save_interval=10)
#
#     # 最后输出剩余结果
#     print(f"验证完成，所有结果已保存到: {output_path}")
#
#     # 输出统计结果
#     print(f"\n统计结果：")
#     print(f"真实新闻数量: {true_count}")
#     print(f"虚假新闻数量: {false_count}")
#     print(f"正确判断数量: {correct_count}")
#     print(f"错误判断数量: {incorrect_count}")
#
#     # 计算并输出准确率，防止除以零
#     total_judgments = correct_count + incorrect_count
#     if total_judgments > 0:
#         accuracy = (correct_count / total_judgments) * 100
#         print(f"准确率: {accuracy:.2f}%")
#     else:
#         print("没有进行任何判断，无法计算准确率。")
#
#
# if __name__ == "__main__":
#     main()
#
#
#断点续训功能
# import json
# import os
# from zhipuai import ZhipuAI
# from tqdm import tqdm  # 导入 tqdm
#
# # 初始化ZhipuAI客户端
# client = ZhipuAI(api_key="37f403ad7e1e00048a738cc98742eae5.2XvzU7T35ArxyUbK")  # 填写您自己的APIKey
#
# def evaluate_news(news_list, output_path, progress_path, save_interval=10):
#     results = []
#     true_count = 0
#     false_count = 0
#     correct_count = 0
#     incorrect_count = 0
#
#     # 尝试读取进度文件
#     start_idx = 0
#     if os.path.exists(progress_path):
#         with open(progress_path, 'r') as f:
#             start_idx = int(f.read().strip())  # 读取已处理的条目数
#         print(f"从第 {start_idx + 1} 条新闻开始处理...")
#
#     # 使用 tqdm 包装 news_list，显示进度条，跳过已处理的新闻
#     for idx, item in enumerate(tqdm(news_list[start_idx:], desc="Evaluating News", unit="news"), start=start_idx):
#         news = item['txt']
#         label = item['label']
#
#         try:
#             # 单个新闻的处理
#             response = client.chat.completions.create(
#                 model="glm-4-plus",  # 填写需要调用的模型编码
#                 messages=[
#                     {"role": "system",
#                      "content": "你是一个网络的新闻打假博主，你会依据你原有的知识和最近的一些新闻，判断一个新闻是真还是假的。此外，输出时请严格按照先说出真假，然后列出理由的形式输出，重要的是你的适应性很强，不需要知道新闻的全部，甚至只需要一段描述性的话语，你也能判断它的真假。如果没有完整的新闻，你就直接按照逻辑推理和一些情感的分析去判断。但是你不用复述以上段句话，你只需要知道这个身份并且严格遵守，发展你的技能吧"},
#                     {"role": "user", "content": news}
#                 ],
#             )
#             result = response.choices[0].message['content'].strip().lower()  # 获取模型输出并处理
#             results.append((news, result))
#
#             # 统计真实和虚假新闻
#             if label:
#                 true_count += 1  # 实际为真实
#                 if result == "真实":  # 假设模型输出"真实"表示正确判断
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1
#             else:
#                 false_count += 1  # 实际为虚假
#                 if result == "虚假":  # 假设模型输出"虚假"表示正确判断
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1
#         except Exception as e:
#             # 捕获异常，确保程序不中断
#             results.append((news, f"Error processing news: {str(e)}"))
#
#         # 每隔 `save_interval` 条新闻保存一次结果和进度
#         if (idx + 1) % save_interval == 0:
#             save_results_to_file(results, output_path)
#             save_progress_to_file(idx + 1, progress_path)
#             print(f"已保存前 {idx + 1} 条新闻的评估结果和进度到: {output_path} 和 {progress_path}")
#
#     # 处理完所有新闻后，保存最终结果和进度
#     save_results_to_file(results, output_path)
#     save_progress_to_file(len(news_list), progress_path)
#     print(f"已保存所有新闻的评估结果和进度。")
#
#     return results, true_count, false_count, correct_count, incorrect_count
#
# def read_news_from_file(file_path):
#     news_list = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             news_list.append(json.loads(line))  # 逐行读取 JSONL 数据并解析
#     return news_list
#
# def save_results_to_file(results, output_path):
#     # 以追加模式保存结果，避免覆盖之前的数据
#     with open(output_path, 'a', encoding='utf-8') as file:
#         for news, result in results:
#             file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')  # 将每条新闻和结果写入文件
#
# def save_progress_to_file(progress, progress_path):
#     # 保存当前的进度（处理了多少条新闻）
#     with open(progress_path, 'w') as f:
#         f.write(str(progress))
#
# def main():
#     # 指定文件路径
#     file_path = "C:/Users/lk/Desktop/GLM4/data/politifact_test.jsonl"  # 请将此处替换为你的文件路径
#     output_path = "C:/Users/lk/Desktop/GLM4/data/evaluation_results.jsonl"  # 输出结果文件路径
#     progress_path = "C:/Users/lk/Desktop/GLM4/data/progress.txt"  # 保存进度的文件路径
#
#     # 从文件读取新闻数据
#     news_dataset = read_news_from_file(file_path)
#
#     # 对新闻数据集进行批量测评，每 10 条保存一次结果和进度
#     results, true_count, false_count, correct_count, incorrect_count = evaluate_news(news_dataset, output_path, progress_path, save_interval=10)
#
#     # 最后输出剩余结果
#     print(f"验证完成，所有结果已保存到: {output_path}")
#
#     # 输出统计结果
#     print(f"\n统计结果：")
#     print(f"真实新闻数量: {true_count}")
#     print(f"虚假新闻数量: {false_count}")
#     print(f"正确判断数量: {correct_count}")
#     print(f"错误判断数量: {incorrect_count}")
#
#     # 计算并输出准确率，防止除以零
#     total_judgments = correct_count + incorrect_count
#     if total_judgments > 0:
#         accuracy = (correct_count / total_judgments) * 100
#         print(f"准确率: {accuracy:.2f}%")
#     else:
#         print("没有进行任何判断，无法计算准确率。")
#
# if __name__ == "__main__":
#     main()
#
#
# import json
# import os
# from zhipuai import ZhipuAI
# from tqdm import tqdm  # 导入 tqdm
#
# # 初始化ZhipuAI客户端
# client = ZhipuAI(api_key="37f403ad7e1e00048a738cc98742eae5.2XvzU7T35ArxyUbK")  # 填写您自己的APIKey
#
# def evaluate_news(news_list, output_path, progress_path, save_interval=10):
#     results = []
#     true_count = 0
#     false_count = 0
#     correct_count = 0
#     incorrect_count = 0
#
#     # 尝试读取进度文件
#     start_idx = 0
#     if os.path.exists(progress_path):
#         with open(progress_path, 'r') as f:
#             start_idx = int(f.read().strip())  # 读取已处理的条目数
#         print(f"从第 {start_idx + 1} 条新闻开始处理...")
#
#     # 使用 tqdm 包装 news_list，显示进度条，跳过已处理的新闻
#     for idx, item in enumerate(tqdm(news_list[start_idx:], desc="Evaluating News", unit="news"), start=start_idx):
#         news = item['txt']
#         label = item['label']
#
#         try:
#             # 单个新闻的处理
#             response = client.chat.completions.create(
#                 model="glm-4-plus",  # 填写需要调用的模型编码
#                 messages=[
#                     {"role": "system",
#                      "content": "你是一个网络的新闻打假博主，你会依据你原有的知识和最近的一些新闻，判断一个新闻是真还是假的。此外，输出时请严格按照先说出真假，然后列出理由的形式输出，重要的是你的适应性很强，不需要知道新闻的全部，甚至只需要一段描述性的话语，你也能判断它的真假。如果没有完整的新闻，你就直接按照逻辑推理和一些情感的分析去判断。但是你不用复述以上段句话，你只需要知道这个身份并且严格遵守，发展你的技能吧。输出格式：先输出一个结果：真实/虚假，然后再理由/思考链："},
#                     {"role": "user", "content": news}
#                 ],
#             )
#             result = response.choices[0].message.content.strip().lower()  # 修改这行代码
#             results.append((news, result))
#
#             # 统计真实和虚假新闻
#             # if label:
#             #     true_count += 1  # 实际为真实
#             #     if result == "真实":  # 假设模型输出"真实"表示正确判断
#             #         correct_count += 1
#             #     else:
#             #         incorrect_count += 1
#             # else:
#             #     false_count += 1  # 实际为虚假
#             #     if result == "虚假":  # 假设模型输出"虚假"表示正确判断
#             #         correct_count += 1
#             #     else:
#             #         incorrect_count += 1
#             # 初始化保存结果的列表
#             data = []
#
#             # 打开并读取 jsonl 文件
#             with open('C:/Users/lk/Desktop/GLM4/data/politifact_test.jsonl', 'r') as file:
#                 for line in file:
#                     # 将每一行转换为 Python 字典
#                     item = json.loads(line)
#
#                     # 检查 label 的值，并将其保存到 data 列表中
#                     if 'label' in item:
#                         label_value = item['label']
#                         if label_value:
#                             true_count += 1  # 实际为真实
#                             print('True')
#                             data.append(True)
#                             if result == "真实":  # 假设模型输出"真实"表示正确判断
#                                  correct_count += 1
#                             else:
#                                  incorrect_count += 1
#                         else:
#                             false_count += 1  # 实际为虚假
#                             print('False')
#                             data.append(False)
#                             if result == "虚假":  # 假设模型输出"虚假"表示正确判断
#                                 correct_count += 1
#                             else:
#                                 incorrect_count += 1
#                     else:
#                         data.append(None)  # 如果没有 label 字段，保存 None
#         except Exception as e:
#             # 捕获异常，确保程序不中断
#             results.append((news, f"Error processing news: {str(e)}"))
#
#         # 每隔 `save_interval` 条新闻保存一次结果和进度
#         if (idx + 1) % save_interval == 0:
#             save_results_to_file(results, output_path)
#             save_progress_to_file(idx + 1, progress_path)
#             print(f"已保存前 {idx + 1} 条新闻的评估结果和进度到: {output_path} 和 {progress_path}")
#
#     # 处理完所有新闻后，保存最终结果和进度
#     save_results_to_file(results, output_path)
#     save_progress_to_file(len(news_list), progress_path)
#     print(f"已保存所有新闻的评估结果和进度。")
#
#     return results, true_count, false_count, correct_count, incorrect_count
#
# def read_news_from_file(file_path):
#     news_list = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             news_list.append(json.loads(line))  # 逐行读取 JSONL 数据并解析
#     return news_list
#
# def save_results_to_file(results, output_path):
#     # 以追加模式保存结果，避免覆盖之前的数据
#     with open(output_path, 'a', encoding='utf-8') as file:
#         for news, result in results:
#             file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')  # 将每条新闻和结果写入文件
#
# def save_progress_to_file(progress, progress_path):
#     # 保存当前的进度（处理了多少条新闻）
#     with open(progress_path, 'w') as f:
#         f.write(str(progress))
#
# def main():
#     # 指定文件路径
#     file_path = "C:/Users/lk/Desktop/GLM4/data/dev_test.jsonl"  # 请将此处替换为你的文件路径
#     output_path = "C:/Users/lk/Desktop/GLM4/outputs/evaluation_results4.jsonl"  # 输出结果文件路径
#     progress_path = "C:/Users/lk/Desktop/GLM4/outputs/progress4.txt"  # 保存进度的文件路径
#
#     # 从文件读取新闻数据
#     news_dataset = read_news_from_file(file_path)
#
#     # 对新闻数据集进行批量测评，每 10 条保存一次结果和进度
#     results, true_count, false_count, correct_count, incorrect_count = evaluate_news(news_dataset, output_path, progress_path, save_interval=10)
#
#     # 最后输出剩余结果
#     print(f"验证完成，所有结果已保存到: {output_path}")
#
#     # 输出统计结果
#     print(f"\n统计结果：")
#     print(f"真实新闻数量: {true_count}")
#     print(f"虚假新闻数量: {false_count}")
#     print(f"正确判断数量: {correct_count}")
#     print(f"错误判断数量: {incorrect_count}")
#
#     # 计算并输出准确率，防止除以零
#     total_judgments = correct_count + incorrect_count
#     if total_judgments > 0:
#         accuracy = (correct_count / total_judgments) * 100
#         print(f"准确率: {accuracy:.2f}%")
#     else:
#         print("没有进行任何判断，无法计算准确率。")
#
# if __name__ == "__main__":
#     main()
#
#
#
import json
import os
from zhipuai import ZhipuAI
from tqdm import tqdm

# 初始化ZhipuAI客户端
client = ZhipuAI(api_key="37f403ad7e1e00048a738cc98742eae5.2XvzU7T35ArxyUbK")  # 填写您自己的APIKey

# def evaluate_news(news_list, output_path, progress_path, save_interval=10):
#     results = []
#     true_count = 0
#     false_count = 0
#     correct_count = 0
#     incorrect_count = 0
#
#     # 尝试读取进度文件
#     start_idx = 0
#     if os.path.exists(progress_path):
#         with open(progress_path, 'r') as f:
#             start_idx = int(f.read().strip())  # 读取已处理的条目数
#         print(f"从第 {start_idx + 1} 条新闻开始处理...")
#     data = []
#
#     # 使用 tqdm 包装 news_list，显示进度条，跳过已处理的新闻
#     for idx, item in enumerate(tqdm(news_list[start_idx:], desc="Evaluating News", unit="news"), start=start_idx):
#         news = item['txt']
#         label = item['label']  # 确保这是 'true' 或 'false' 字符串
#
#         try:
#             # 单个新闻的处理
#             response = client.chat.completions.create(
#                 model="glm-4-plus",  # 填写需要调用的模型编码
#                 messages=[
#                     {"role": "system",
#                      "content": "你是一个新闻打假博主，会依据知识判断新闻是真还是假的。输出格式为：真实/虚假。"},
#                     {"role": "user", "content": news}
#                 ],
#             )
#             result = response.choices[0].message.content.strip().lower()  # 确保结果格式一致为小写'真实'或'虚假'
#             results.append((news, result))
#
#             # 统计真实和虚假新闻
#             if result == "真实":
#                 correct_count += 1
#                 data.append(True)
#             elif result == "虚假":
#                 incorrect_count += 1
#                 data.append(False)
#
#         except Exception as e:
#             # 捕获异常，确保程序不中断
#             results.append((news, f"Error processing news: {str(e)}"))
#
#     # 统计所有新闻的数量
#         # 每隔 `save_interval` 条新闻保存一次结果和进度
#         if (idx + 1) % save_interval == 0:
#             save_results_to_file(results, output_path)
#             save_progress_to_file(idx + 1, progress_path)
#             print(f"已保存前 {idx + 1} 条新闻的评估结果和进度到: {output_path} 和 {progress_path}")
#
#     # 处理完所有新闻后，保存最终结果和进度
#     save_results_to_file(results, output_path)
#     save_progress_to_file(len(news_list), progress_path)
#     print(f"已保存所有新闻的评估结果和进度。")
#
#     return results, true_count, false_count, correct_count, incorrect_count
def evaluate_news(news_list, output_path, progress_path, save_interval=10):
    results = []  # 存储所有处理结果
    true_count = 0  # 统计‘真实’新闻数量
    false_count = 0  # 统计‘虚假’新闻数量
    correct_count = 0  # 标记为‘真实’且正确的新闻数量
    incorrect_count = 0  # 标记为‘虚假’且错误的新闻数量

    # 尝试读取进度文件
    start_idx = 0  # 默认从0开始
    if os.path.exists(progress_path):  # 检查是否有进度文件
        with open(progress_path, 'r') as f:
            start_idx = int(f.read().strip())  # 从进度文件中读取已处理的条目数
        print(f"从第 {start_idx + 1} 条新闻开始处理...")

    data = []  # 用于存储真假标签

    # 使用 tqdm 包装 news_list，显示进度条，跳过已处理的新闻
    for idx, item in enumerate(tqdm(news_list[start_idx:], desc="Evaluating News", unit="news"), start=start_idx):
        news = item['txt']  # 提取新闻内容
        label = item['label']  # 提取新闻的真假标签

        try:
            # 调用 AI 模型进行新闻的真假评估
            response = client.chat.completions.create(
                model="glm-4-plus",  # 调用的模型
                messages=[
                    {"role": "system",
                     "content": "你是一个新闻打假博主，会依据知识判断新闻是真还是假的。输出格式为：真实/虚假,理由：。"},
                    {"role": "user", "content": news}
                ],
            )
            result = response.choices[0].message.content.strip().lower()  # 获取模型返回的结果并转换为小写
            first_two_chars = result[:2]
            print(first_two_chars)
            results.append((news, result))  # 将新闻和结果追加到结果列表

            # 统计真实和虚假新闻的数量
            if result == "真实":
                correct_count += 1  # 正确的真实新闻计数+1
                data.append(True)  # 真实新闻存储为 True
            elif result == "虚假":
                incorrect_count += 1  # 错误的虚假新闻计数+1
                data.append(False)  # 虚假新闻存储为 False

        except Exception as e:
            # 捕获异常，确保程序不中断
            results.append((news, f"Error processing news: {str(e)}"))

        # 每隔 `save_interval` 条新闻保存一次结果和进度
        if (idx + 1) % save_interval == 0:
            save_results_to_file(results, output_path)  # 调用保存函数保存结果
            save_progress_to_file(idx + 1, progress_path)  # 调用保存函数保存进度
            print(f"已保存前 {idx + 1} 条新闻的评估结果和进度到: {output_path} 和 {progress_path}")

    # 处理完所有新闻后，保存最终结果和进度
    save_results_to_file(results, output_path)  # 保存最终结果
    save_progress_to_file(len(news_list), progress_path)  # 保存最终进度
    print(f"已保存所有新闻的评估结果和进度。")
    # 返回处理结果和统计数据
    return results, true_count, false_count, correct_count, incorrect_count


def read_news_from_file(file_path):
    news_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            news_list.append(json.loads(line))  # 逐行读取 JSONL 数据并解析
    return news_list

# def save_results_to_file(results, output_path):
#     # 以追加模式保存结果，避免覆盖之前的数据
#     with open(output_path, 'a', encoding='utf-8') as file:
#         for news, result in results:
#             file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')  # 将每条新闻和结果写入文件
#
# 复写去重test1
# def save_results_to_file(results, output_path):
#     # 以追加模式保存结果，避免覆盖之前的数据
#     with open(output_path, 'a', encoding='utf-8') as file:
#         seen = set()  # 用于记录已写入的新闻
#         for news, result in results:
#             if news not in seen:  # 检查新闻是否已写入
#                 file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')
#                 seen.add(news)  # 将新闻添加到已写入集合
#当前批次重复t2
def save_results_to_file(results, output_path,dev_output_path):
    # 检查是否已有保存的文件
    seen = set()

    # 读取已保存的新闻，避免重复
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            for line in file:
                saved_item = json.loads(line)
                seen.add(saved_item['news'])  # 将已有的新闻添加到已见集合中

    # 用于当前批次内去重
    current_batch_seen = set()

    # 以追加模式保存新的结果
    with open(output_path, 'a', encoding='utf-8') as file:
        for news, result in results:
            if news not in seen and news not in current_batch_seen:  # 同时避免跨批次和当前批次的重复
                file.write(json.dumps({"news": news, "result": result}, ensure_ascii=False) + '\n')
                seen.add(news)  # 将新写入的新闻添加到已见集合
                current_batch_seen.add(news)  # 将当前批次的新闻添加到当前批次集合
    with open(dev_output_path, 'a', encoding='utf-8') as file:
        for news, first_two_chars in results:
            file.write(json.dumps({"news": news, "first_two_chars": first_two_chars}, ensure_ascii=False) + '\n')

def save_progress_to_file(progress, progress_path):
    # 保存当前的进度（处理了多少条新闻）
    with open(progress_path, 'w') as f:
        f.write(str(progress))

def main():
    # 指定文件路径
    file_path = "C:/Users/lk/Desktop/GLM4/data/dev_test.jsonl"  # 替换为上传的文件路径
    output_path = "C:/Users/lk/Desktop/GLM4/outputs/evaluation_results4.jsonl"  # 输出结果文件路径
    progress_path = "C:/Users/lk/Desktop/GLM4/outputs/progress4.txt"  # 保存进度的文件路径
    dev_output_path="C:/Users/lk/Desktop/GLM4/outputs/dev_output.txt"
    # 从文件读取新闻数据
    news_dataset = read_news_from_file(file_path)

    # 对新闻数据集进行批量测评，每 10 条保存一次结果和进度
    results, true_count, false_count, correct_count, incorrect_count = evaluate_news(news_dataset, output_path, progress_path, save_interval=10)

    # 最后输出剩余结果
    print(f"验证完成，所有结果已保存到: {output_path}")

    # 输出统计结果
    print(f"\n统计结果：")
    print(f"真实新闻数量: {true_count}")
    print(f"虚假新闻数量: {false_count}")
    print(f"正确判断数量: {correct_count}")
    print(f"错误判断数量: {incorrect_count}")

    # 计算并输出准确率，防止除以零
    total_judgments = correct_count + incorrect_count
    if total_judgments > 0:
        accuracy = (correct_count / total_judgments) * 100
        print(f"准确率: {accuracy:.2f}%")
    else:
        print("没有进行任何判断，无法计算准确率。")

if __name__ == "__main__":
    main()



