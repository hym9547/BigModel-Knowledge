### 大模型：

- 使用：加载->处理数据->返回结果
- 微调：量化/lora ->加载 ->数据集->合适的微调方法->保存

### requirements.txt生成：

```python
pip install pipreqs
pipreqs natural-sql-7b --encoding=utf-8 --force
```



# 数据
### 数据加载器
##### 1、 torch的DataLoader 
```python
from torch.utils.data import DataLoader
```
```python
train_dataloader = DataLoader(
                    dataset = train_examples,
                    shuffle=True,
                    batch_size=BATCH_SIZE
                    )
```
参数：
- dataset 数据集
- shuffle 乱序
- batch_size 单词获取数据 

备注：
- 作用：直接加载数据使用，train_dataloader直接交给train函数调用
- 特点：适用于处理完成后的数据

##### 2、transformer的DataLoader 
 ```python
from datasets import load_dataset
 ```
```python
datasets = load_dataset("json", data_files=self.data_path)
```
参数：
- path 本地路径或huggingface数据集名称
- name 数据集子集（版本/名称）
- data_files 执行数据集文件
- split 切割数据集

备注：
- 作用：默认加载结果为DatasetDict；如果未指定split，那么默认全部加载train部分，指定其他部分需要格式对应；指定后的加载直接为Dataset
- 特点：适用于需要进行二次操作的数据（embedding、padding等）
- 切割：加载好的数据集直接调用train_test_split

### 数据处理
##### 1、pandas
- ```pd.read_csv() / pd.read_excel()``` ：用于从 CSV 文件或 Excel 文件中读取数据，并创建一个 DataFrame。
- ```df.head(n) / df.tail(n)```： 显示 DataFrame 的前 n 行或后 n 行，默认显示前5行或后5行。
- ```df.loc[] / df.iloc[]```： 通过标签或位置进行数据的选择。loc 主要用于基于标签的选择，而 iloc 则是基于整数位置的选择。
- ```df.query()```：使用字符串表达式过滤 DataFrame 中的行。
- ```df.filter()```：根据列名或索引筛选行或列。
- ```df.dropna()```：删除包含缺失值的行或列。
- ```df.fillna(value)```：用指定值填充缺失数据。
- ```df.replace(old, new)```：替换 DataFrame 中的值。
- ```df.duplicated()```：查找重复的行。
- ```df.drop_duplicates()```：删除重复的行。
- ```df.apply(func)```：对 DataFrame 的每一行或每一列应用函数。
- ```df.groupby('column').agg(func)```：按照某一列分组并执行聚合操作，如求和、平均值等。
- ```df.merge()```：类似于 SQL 中的 JOIN 操作，可以将两个 DataFrame 进行合并。
- ```df.pivot_table()```：创建一个电子表格风格的透视表作为 DataFrame。
- ```df.sort_values(by='column')```：根据一列或多列的值对 DataFrame 进行排序。
- ```df.nlargest(n, 'column') / df.nsmallest(n, 'column')```：获取某列中最大的 n 个值或最小的 n 个值对应的行。
- ```df.sample(n)```：随机抽取 n 行。

# 模型
### 加载
##### 1、embedding模型加载 
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_PATH)
```
参数：
- model_path 模型地址或huggingface的模型名称
- device cuda或cpu

##### 2、大语言模型加载 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
        self.model_path,  
        device_map="auto", # 自动选择设备
        torch_dtype=torch.float16, # 计算精度
        low_cpu_mem_usage=True,  # 减少CPU内存占用 
        offload_folder="offload",  # 临时卸载路径
        quantization_config=q4, # 8bit量化
)
```
参数：
 - ```pretrained_model_name_or_path```： 模型地址或huggingface的名称
 - ```**kwargs```： 解析键值对参数

```python
tokenizer = AutoTokenizer.from_pretrained(self.model_path)
```
参数：
 - ```pretrained_model_name_or_path```： 模型地址或huggingface的名称
 - ```**kwargs```： 解析键值对参数

##### 3、量化方法-模型加载参数
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
q8 = BitsAndBytesConfig(
        load_in_4bit=False, # 4bit
        load_in_8bit=True, # 8bit
        bnb_4bit_use_double_quant=False # 4bit加倍
        )
q4 = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit
        bnd_4bit_use_double_quant = False, # 双量化
        bnd_4bit_quant_type = "nf4" # 量化方式
        )
prepare_model = prepare_model_for_kbit_training(model) # 修复模型量化后可能造成训练的不稳定；不训练则不用执行
```


### 微调
##### 1、SentenceTransformer封装的fit方法
```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
train_loss = losses.ContrastiveLoss(model=model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4, # 3-5次
    show_progress_bar=True,
    warmup_steps=100,
    optimizer_params={'lr': 2e-5},  # 设置学习率 
    evaluator=EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='val'), 
    evaluation_steps=1000,  
    save_best_model=True,   # 保存最佳模型
    output_path=SAVE_PATH
)
```
参数：
- ```train_objectives```： 数据加载器、损失函数联合列表
- ```epochs```: 代数
- ```show_progress_bar```: 显示训练进度条
- ```warmup_steps``` 学习率预热步数；避免一开始学习率太高而跳跃；通常为评估部署的5%-10%
- ```optimizer_params```: 优化器参数；lr表示学习率，一般设置2e-5
- ```evaluator```: 评估器->决定最佳模型
- ```evaluation_steps```: 多少步验证性能
- ```save_best_model```: 保存最佳模型
- ```output_path```: 保存模型的路径

备注：
- 特点：完全封装，只需要一些参数就可以触发训练操作
- 作用：适用于embedding的微调

##### 2、transformers的Trainer
```python
from transformers import TrainingArguments
    training_args = TrainingArguments(**kwargs)
```
```python
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
)
trainer.train()
```
参数：
- ```model```: 训练模型
- ```args```: 参数配置 （学习率、地址、数据等）
- ```train_dataset```: 训练集
- ```train_dataset```: 验证集
- ```data_collator```: 标准数据集加载工具

备注：
- 特点：标准化、组件化，方便调用与各阶段调试
- 作用：适用于对话等生成大模型微调

##### 3、lora方法-模型调试
```ptyhon 
from peft import LoraConfig, get_peft_model
loraconfig = LoraConfig(
    r=8, # 矩阵
    lora_alpha=16,  #
    target_modules=['q_proj', 'v_proj'], # 调整层
    lora_dropout = 0.05, # 随机丢失
    bias = 'none', # 偏置层
    task_type='CAUSAL_LM' # 任务类型 
)
lora_model = get_peft_model(prepare_model, loraconfig) # 合并lora参数与模型 
lora_model.base_model.config.use_cache = False  # 防止训练过程产生冲突
```
参数：
- ```r```：矩阵因子（秩）；决定更新矩阵的维度大小；代表训练模型的大小
- ```lora_alpha```：缩放因子，调节lora矩阵对模型的影响程度
- ```target_modules```：调整目标层次；一般为['q_proj', 'v_proj']注意力层次的QV
- ```lora_dropout```：dropout同理，随机丢失比列
- ```bias```：偏置层是否调整
- ```task_type```：需要调整的模型类别（对话、分类等）

### 使用
##### 1、生成式模型
```python
from transformers import pipeline
model_pipeline = pipeline(
            task ='text-generation',
            model = model,
            tokenizer = tokenizer,
            max_new_tokens=150,
            temperature=0.01,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
            )
prompt='''   '''
result = model_pipeline(question)[0]['generated_text']  # question可跟prompt结合（类似于对问题添加了很多的描述/备注/资料等，增强模型的理解能力）
```
##### 2、embedding模型
```python
embedding = model.encode(text, normalize_embeddings=True) # 默认返回numpy数组
```
扩展：embedding对文本处理后存入milvus数据库
```python
collection = Collection(coll_name) # 连接milvus数据库集合
collection.load() # 加载到内存
entities = [
    ids,
    texts,
    embeddings,
    updata_tims
]  # 将embedding添加到需要存入milvus的格式数据中，匹配列
result = collection.insert(entities) # 返回插入结果；插入没有数据已存在一说（设置了主键ID），会直接更新该条数据
collection.flush() # 永久化，从内存保存到磁盘
```
