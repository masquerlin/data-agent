from openai import OpenAI
from typing import List, Optional, Tuple
import json
# from config import opeani_api_keys
import pandas as pd
from auto_plot_mat import auto_plot
from pandasql import sqldf
#加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from tabulate import tabulate
import gradio as gr
import argparse
def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--openai_key", type=str,
                        default="", help="openai-keys")
        args = parser.parse_args()
        return args
args = get_args()
openai_key = args.openai_key
temperature=0.2
History = List[Tuple[str, str]]
def process_file(file):
    if not isinstance(file, list):
        file = [file]
    df = pd.DataFrame()
    for i in range(len(file)):
        data_temp = {'new_df' + str(i): {'type': file[i].name, 'description': '-'*10}}
        df_temp = pd.DataFrame.from_dict(data_temp, orient='index').reset_index()
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
        df_real = pd.read_csv(file[i].name)
        
        schema_info = generate_schema(df_real)
        df_schema = pd.DataFrame.from_dict(schema_info, orient='index').reset_index()
        df = pd.concat([df, df_schema], axis=0, ignore_index=True)
    return df


def generate_schema(df_real):
    df_string = df_real.head().to_string()
    prompt = """
    请你根据一个dataframe的前5行展示的信息，来写一个它的schema_info，schema_info是一个json下面是一个示例：

    ########################
    dataframe的前5行:
            Customer ID  Age  Gender Marital Status    Education Level
    0        84966   23  female        married   associate degree
    1        95568   26    male        widowed          doctorate
    2        10544   29  female         single   associate degree
    3        77033   20    male       divorced  bachelor's degree
    4        88160   25  female      separated  bachelor's degree
    schema_info:
    {
        'Customer ID': {'type': 'int', 'description': 'Unique identifier for each customer'},
        'Age': {'type': 'int', 'description': 'The age of the customer'},
        'Gender': {'type': 'str', 'description': 'The gender of the customer'},
        'Marital Status': {'type': 'str', 'description': 'The marital status of the customer'},
        'Education Level': {'type': 'str', 'description': 'The highest level of education achieved by the customer'}
    }
    ########################
    dataframe的前5行:
    """ + df_string + '\n' + 'schema_info:'
    schema_info = run_llm(prompt)
    try:
        result = json.loads(schema_info.replace("'", "\""))
    except Exception as e:
        result = {}
        column_types = df_real.dtypes
        for col in df_real.columns:
            result[col] = {
                "type": str(column_types[col]),
                "description": str(df_real[col][0])  # 使用第一行的数据作为描述
            }
        print(f"a error occured when transfer the schema_info string to dict: {e}, it will use the row's data to be the description")
    return result



def clear_session() -> History:
    return '', []
def generate_sql(query, table_name, schema, history = [], debug = False, error =''):
    if not debug:
        prompt_prefix = '''
        您是一个sqlalchemy专家。请根据以下的表名和相应的column生成与给定问题相对应的SQL语句, SQL语句可以从一个表或者多个表选择数据，可以用join或者子查询比较等等方法, SQL语句不要带有换行符, 表名不能带单引号,如df 就是 df ,
        强调:每个schema的引用都要加上单引号, 如'schema'。
        
        下面是问题和对应的SQL语句示例:
        ########################
        问题: 本科学历的人的平均工资是多少？
        SQL语句:  SELECT avg(df.'Income Level') from df where df.'Education Level' like '%bachelor%';
        
        问题: 不同地区的平均工资是多少？
        SQL语句: SELECT df.'Geographic Information',avg(df.'Income Level') from df group by df.'Geographic Information';

        问题: 请你单身的人数有多少？
        SQL语句: SELECT count('Customer ID') from df where df.'Marital Status' like '%single%';;
        ########################
        
        问题: {}
        表名：{}
        schema: {}
        
        SQL语句:     
        '''
        # table_name = list(locals().keys())[list(locals().values()).index(df.items())]

        prompt_content = prompt_prefix.format(query, table_name, schema)
        answer = run_llm(prompt_content)
        history.append({'role':'user',"content":prompt_content})
        history.append({'role':'assistant',"content":answer})
        
        print(f'sql_answer******************{answer}')
        return answer, history
    else:
        prompt_debug = """之前的sql生成语句:{}, 报错了{},请重新生成sql语句,回答要是纯的sql语句:
        """
        sql_before = error.split('...')[1]
        e = error.split('...')[0]
        prompt_debug_content = prompt_debug.format(sql_before, e)
        answer_debug = run_llm(prompt_debug_content, history=history)
        history.append({'role':'user',"content":prompt_debug_content})
        history.append({'role':'assistant',"content":answer_debug})
        return answer_debug, history

def run_llm(query, openai_key=openai_key, api_base='', engine='',history = []):
    
    client = OpenAI(
            api_key=openai_key,  # this is also the default, it can be omitted
        )
    
    messages = [{"role":"system","content":"You are an useful AI assistant that helps people solve the problem step by step."}]
    messages.extend(history)
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
    engine="gpt-3.5-turbo"
    response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=0.5,
                    # max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)

    result = response.choices[0].message.content
    return result

def model_chat(query: Optional[str], history: Optional[History], df_process):
    split_indices = df_process.index[df_process['description'] == '-'*10].tolist()
    paths = []
    
    # 如果没有分隔符，则只有一个DataFrame
    if not split_indices:
        dfs = [df_process]
    else:
        dfs = []
        start_idx = 0
        for end_idx in split_indices:
            if end_idx != 0:
                dfs.append(df_process.iloc[start_idx:end_idx])
            start_idx = end_idx + 1
            paths.append(df_process.loc[end_idx, 'type'])
        dfs.append(df_process.iloc[start_idx:])
    # 打印每个DataFrame
    table = []
    scema = ''
    for i, d in enumerate(dfs):
        df_name_str = "df_use_" + str(i+1)
        table.append(df_name_str)
        locals()[df_name_str] = pd.read_csv(paths[i])
        scema_temp = d.set_index('index').to_dict(orient='index')
        scema += df_name_str + '的schema:' + '\n\n' + str(scema_temp) + '\n\n'
    table_name = ','.join(table)
    sql_statement, sql_history = generate_sql(query, table_name, scema)
    print(f"sql_statement**********************{sql_statement}")
    i = 0
    while i < 3:
        print(f"i_______________________{i}")
        try:
            pandaSQL_solution = sqldf(sql_statement, locals())
            break
        except Exception as e:
            print(f"there're bugs in sql, debuging by the gpt at the {i+1} time......")
            e = str(e) + '...' + str(sql_statement)
            sql_statement, sql_history = generate_sql(query, table_name, scema, history=sql_history, debug=True, error = e)
            print(f"{i+1} time fixed sql:{sql_statement}")
        i+=1

    
    image_arrays = auto_plot(df=pandaSQL_solution)

    mylist = list()
    mylist.append(query)
    prompt_list = [query]
    string_data = pandaSQL_solution.to_string()
    prompt_list.append(sql_statement)
    prompt_list.append(string_data)
    prompt_report = '''现在有一个列表，列表的第一个值是问题，第二个值是用大模型实现的text to sql, 第三个值是根据sql查出来的数据，请你根据问题和查出来的数据，形成一个数据分析报告进行回答，这个分析报告要详细清晰，有逻辑和条理:{}'''.format(str(prompt_list))
    report_answer = run_llm(query=prompt_report)
    mylist.append(report_answer)
    responses = list()
    if len(history) > 0:
        for history_msg in history:
            responses.append(history_msg)
    responses.append(mylist)
    return "", responses, image_arrays, pandaSQL_solution

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("""<center><font size=10>Data Agent。</center>""")
    with gr.Row(equal_height=True):
            file_output = gr.Files()
            df_process = gr.DataFrame()
            file_output.upload(process_file, inputs= [file_output],outputs=[df_process])
    gallery = gr.Gallery(label="最终的结果图片").style(height='auto',columns=4)
    df = gr.outputs.Dataframe(type='pandas')
    chatbot = gr.Chatbot(label='报告展示')
    textbox = gr.Textbox(lines=2, label='请描述您想查询的数据')
    
    
    with gr.Row():
        with gr.Column(scale=2):
            clear_history = gr.Button("🧹 清除历史对话")
        with gr.Column(scale=2):
            sumbit = gr.Button("🚀 发送")
    
    sumbit.click(model_chat, [textbox, chatbot, df_process], [textbox, chatbot, gallery, df])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot])
    

    

# demo.queue(api_open=False).launch(max_threads=10, height=800, share=False, server_name="0.0.0.0", server_port=8888)
demo.queue(api_open=False).launch(max_threads=10, height=800, share=False)