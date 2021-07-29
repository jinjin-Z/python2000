#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import datetime

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

print('args: ', sys.argv)
config = json.loads(sys.argv[1])




conf = SparkConf()
sc = SparkContext(conf=conf)
sql_context = HiveContext(sc)

columns = [


def to_array(col):
    """将Vector类型的列转换为Array类型

    Args:
        col: Vector类型的列名

    Returns:
        一个新的Array类型的列
    """
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)


# 特征预处理,将处理列转换为doubletype
class preprocessor(Transformer):
    def __init__(self):
        super(preprocessor, self).__init__()
    def _transform(self, df):
        to_double = udf(float, DoubleType())
        df = apply_func_to_columns(df,df.columns[1:-1],to_double)
        return df



def top_percents_lift(dataset, positive_prob_col, label_col):
    """计算数据集中预测分数 Top 10% 的样本的 lift

    Args:
        dataset:            DataFrame类型
        positive_prob_col:  正样本概率所在的列名
        label_col:          真实标签所在的列名

    Returns:
        lift
    """
    total_count = dataset.count()
    total_positive_count = dataset.filter(label_col+'> 0.5').count()
    top_records = dataset.orderBy(desc(positive_prob_col)).take(int(total_count/10))
    top_records_count = len(top_records)
    top_positive_count = len([r for r in top_records if r[label_col] > 0.5])
    return (top_positive_count*1.0/top_records_count) / (total_positive_count*1.0/total_count)
    
def auc(dataset, prob_vec_col, label_col):
    """计算数据集分类效果AUC指标

    Args:
        dataset:        DataFrame类型
        prob_vec_col:   概率向量所在的列
        label_col:      真实标签所在的列

    Returns:
        模型AUC
    """
    evaluator = BinaryClassificationEvaluator(rawPredictionCol=prob_vec_col, labelCol=label_col)
    return evaluator.evaluate(dataset, {evaluator.metricName:'areaUnderROC'})

class ProbExtractor(Transformer):
    def __init__(self, prob_col):
        super(ProbExtractor, self).__init__()
        self.prob_col = prob_col
    def _transform(self, df):
        all_columns = df.columns
        return df.withColumn('prob_array', to_array(col(self.prob_col))).select(
            list(all_columns) + [col('prob_array')[1].alias('prob')])

def apply_func_to_columns(df, columns, func):
    """将函数作用到数据的某些列上面

    Args:
        df:         输入数据，DataFrame类型
        columns:    df中需要做函数变换的列
        func:       变换函数

    Returns:
        函数变换之后新的DataFrame
    """
    for c in columns:
        print(c)
        other_cols = set(df.columns) - set([c])
        df = df.select(*(list(other_cols)+[func(df[c]).alias(c)]))
    return df

def print_evaluation_metric(dataset, name, prob_vec_col, positive_prob_col, label_col):
    """打印出模型评估指标，包括AUC和lift

    Args:
        dataset:            DataFrame类型
        name:               数据集名称
        prob_vec_col:       概率向量所在的列
        positive_prob_col:  正样本概率所在的列
        label_col:          真实标签所在的列

    Returns:
        None
    """
    positive_count = dataset.filter(label_col+'> 0.5').count()
    print(name + ' Samples: %d' % dataset.count())
    print('Positive: %d, Negative: %d' % (positive_count, dataset.count()-positive_count))
    print(name + ' AUC: %.4f' % auc(dataset, prob_vec_col, label_col))
    print(name + ' Top 10%% Lift: %.2f' % top_percents_lift(dataset, positive_prob_col, label_col))

'''
 config = 
{"feature_table":"test.purchase_predict_feature","max_date":"2020-06-20","order_table":"odata_prd.prd_p_go_order","output_table":"test.purchase_predict_table"}
'''

# 创建预测数据集
def create_dataset(config):
    
    # 取订单表
    order_table = config["order_table"]
    
    # 取最大日期
    max_date = config["max_date"]
    min_date = (datetime.datetime.strptime(max_date,"%Y-%m-%d") - datetime.timedelta(days = 30)).strftime("%Y-%m-%d")
    month = max_date[:7]
    year_ago = (datetime.datetime.strptime(max_date,"%Y-%m-%d") - datetime.timedelta(days = 365)).strftime("%Y-%m-%d")

    # 使用cursor 构建数据特征集合
    sql = """
    create table if not exists test.ecmodel_tmp1(
    client_id string comment '客户id',
    buy_ct double comment '购买次数',
    freq double comment '购买频率',
    avg_order_dct double comment '平均每天订单数',
    window1 double comment '第一次购买距最后一单天数',
    orderid_dct double comment '用户下单总数',
    recency double comment '最近一次下单距今天的间隔天数',
    productcode_dct double comment '购买产品总数',
    max_tenure double comment '近一年最大订单间隔',
    min_tenure double comment '近一年最小订单间隔')
    """
    print(sql)
    sqlContext.sql(sql)
    sqlContext.sql("truncate table ecmodel_tmp1")
    sqlContext.sql("drop table if exists %s" %config["feature_table"])

    sql = """
    create table if not exists %(feature_table)s(
    busi_date string comment '日期',
    client_id string comment '客户id',
    buy_ct double comment '购买次数',
    freq double comment '购买频率',
    fugou_prob double comment '近一年历史复购率',
    avg_order_dct double comment '平均每天订单数',
    window1 double comment '第一次购买距最后一单天数',
    orderid_dct double comment '用户下单总数',
    recency double comment '最近一次下单距今天的间隔天数',
    productcode_dct double comment '购买产品总数',
    max_tenure double comment '近一年最大订单间隔',
    min_tenure double comment '近一年最小订单间隔')
    """ %{"feature_table":config["feature_table"]}
    print(sql)
    sqlContext.sql(sql)
    sqlContext.sql("truncate table %s" %config["feature_table"])

    sql = """
    with this as
    (
    SELECT substr(create_at,1,10) as busi_date, client_id, order_id, prod_code FROM %(order_table)s 
    where ((app_id = 'ytj' and status in ('4','5') and action_in in ('sub','apply','exchange','tsub') and prod_type in ('5','7'))
    or (prod_code in ('AISTOCK','AITIMING','ANALYSIS','KLINE','CHIPS','MARGINTRADING','LIMITUP') and action_in = 'sub' and status = '5')
    )
    and substr(create_at,1,10) > '%(min_date)s' and substr(create_at,1,10) < '%(max_date)s')

    -- 训练数据特征
    insert overwrite table ecmodel_tmp1
    select ts.client_id,
    coalesce(t0.buy_ct,0) as buy_ct,
    coalesce(t0.freq,0) as freq,
    coalesce(t0.avg_order_dct,0) as avg_order_dct,
    coalesce(t0.window1,0) as window1,
    coalesce(t0.orderid_dct,0) as orderid_dct,
    coalesce(t0.recency,0) as recency,
    coalesce(t0.productcode_dct,0) as productcode_dct,
    coalesce(t1.max_tenure,0) as max_tenure,
    coalesce(t1.min_tenure,0) as min_tenure
    from
    (select client_id from this group by client_id) ts
    left outer join
    (SELECT
    client_id,
    count(distinct busi_date) as buy_ct,
    datediff(max(busi_date),min(busi_date))/count(distinct busi_date) as freq,
    count(distinct order_id) / datediff(max(busi_date),min(busi_date)) as avg_order_dct,
    datediff(max(busi_date),min(busi_date)) as window1,
    count(distinct order_id) as orderid_dct,
    datediff('%(max_date)s',max(busi_date)) as recency,
    count(distinct prod_code) as productcode_dct
    FROM this
    group by client_id) t0 on ts.client_id = t0.client_id
    left outer join
    (select client_id, max(jiange) as max_tenure, min(jiange) as min_tenure from
    (select client_id, datediff(busi_date, last_date) as jiange from
    (select client_id, busi_date, lead(busi_date) over(partition by client_id order by busi_date desc) as last_date
     from (select client_id,busi_date from this where busi_date > '%(year_ago)s' group by client_id,busi_date)t ) s) ss group by ss.client_id) t1 on ts.client_id = t1.client_id
    """ %{"max_date":max_date,"min_date":min_date,"year_ago":year_ago,"order_table":order_table}
    print(sql)
    sqlContext.sql(sql)

    sql = """
    with this as
    (
    SELECT substr(create_at,1,10) as busi_date, client_id, order_id, prod_code FROM %(order_table)s
    where ((app_id = 'ytj' and status in ('4','5') and action_in in ('sub','apply','exchange','tsub') and prod_type in ('5','7'))
    or (prod_code in ('AISTOCK','AITIMING','ANALYSIS','KLINE','CHIPS','MARGINTRADING','LIMITUP') and action_in = 'sub' and status = '5')
    )
    and substr(create_at,1,10) > '%(min_date)s' and substr(create_at,1,10) < '%(max_date)s'),

    this_a as (
    select client_id, count(distinct busi_date)/365 as fugou_prob from
    (select client_id, b.busi_date from
    (select client_id,concat(date_sub(busi_date,1),',',date_sub(busi_date,2),',',date_sub(busi_date,3),','
    ,date_sub(busi_date,4),',',date_sub(busi_date,5),',',date_sub(busi_date,6),',',date_sub(busi_date,7),','
    ,date_sub(busi_date,8),',',date_sub(busi_date,9),',',date_sub(busi_date,10),',',date_sub(busi_date,11),','
    ,date_sub(busi_date,12),',',date_sub(busi_date,13),',',date_sub(busi_date,14),',',date_sub(busi_date,15))
    as busi_date from
    (select client_id, busi_date from this where busi_date > '%(year_ago)s' group by client_id, busi_date) t) tt
    lateral view explode(split(tt.busi_date,',')) b as busi_date) ttt
    group by ttt.client_id )

    insert overwrite table %(feature_table)s
    select '%(busi_date)s' as busi_date, t1.client_id, t1.buy_ct, t1.freq, coalesce(t2.fugou_prob,0) as fugou_prob, t1.avg_order_dct, t1.window1, t1.orderid_dct, t1.recency,
     t1.productcode_dct, t1.max_tenure, t1.min_tenure
       from ecmodel_tmp1 t1 left outer join this_a t2 on t1.client_id = t2.client_id
    """ %{"busi_date":max_date,"max_date":max_date,"min_date":min_date,"year_ago":year_ago,"feature_table":config["feature_table"],"order_table":order_table}
    print(sql)
    
    sqlContext.sql(sql)




# 读取训练数据
sql_for_train = 'select * from test.ecmodel_train'
train_data = sqlContext.sql(sql_for_train)
print('read train_data done!')

# 读取测试数据
sql_for_test = 'select * from test.ecmodel_dev'
test_data = sqlContext.sql(sql_for_test)

columns = [
 'buy_ct',
 'freq',
 'fugou_prob',
 'avg_order_dct',
 'window1',
 'orderid_dct',
 'recency',
 'productcode_dct',
 'max_tenure',
 'min_tenure',
] 

stages = []

    
# stages.append(preprocessor)

# 将所有特征列拼接为一列
vec_assembler = VectorAssembler(inputCols=columns,outputCol='features')
stages.append(vec_assembler)

# 将label进行StringIndex
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel",handleInvalid='skip')
stages.append(labelIndexer)

# 分类模型
classifier = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10,maxDepth=3)
stages.append(classifier)

# 将概率向量中的正样本概率提取出来
prob_extractor = ProbExtractor('probability')
stages.append(prob_extractor)


# 模型训练
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_data)
training_set = pipeline_model.transform(train_data)
testing_set = pipeline_model.transform(test_data)

training_set.show(1)
training_set.select('client_id', 'probability', 'prob', 'prediction', 'indexedLabel').show()

# 输出模型效果
print('Total Samples: %d' % train_data.count())
print_evaluation_metric(training_set, 'Training', 'probability', 'prob', 'indexedLabel')
print_evaluation_metric(testing_set, 'Testing', 'probability', 'prob', 'indexedLabel')


# 创建预测数据集
create_dataset(config)



# 模型预测
predict_set = sql_context.sql('select * from %s' % config["feature_table"])
print('predict_set count: %d' % predict_set.count())
predict_set = pipeline_model.transform(predict_set)
print('predict_set count: %d' % predict_set.count())
predict_set.show(1)

# 预测结果保存到Hive
sql_context.sql('DROP TABLE IF EXISTS %s' % config["output_table"])
predict_set.select('cust_pty_no', 'prob', 'prediction').registerTempTable('purchase_prediction')
sql = 'CREATE TABLE %s AS SELECT * FROM purchase_prediction' % config["output_table"]
sql_context.sql(sql)
print('save prediction to hive done')