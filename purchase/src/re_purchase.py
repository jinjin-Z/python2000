#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator



sc = SparkContext.getOrCreate()
sqlContext = HiveContext(sc)

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



# 读取训练数据
sql_for_train = 'select * from test.ecmodel_train'
train_data = sqlContext.sql(sql_for_train)
print('read train_data done!')

# 读取测试数据
sql_for_test = 'select * from test.ecmodel_dev'
test_data = sqlContext.sql(sql_for_test)


stages = []

# 特征预处理
def preprocessor(df):
    # 将数据类型转换为doubletype
    to_double = udf(float, DoubleType())
    df = apply_func_to_columns(df,df.columns[1:-1],to_double)
    return df
    
stages.append(preprocessor)

# 将所有特征列拼接为一列
vec_assembler = VectorAssembler(inputCols=df.columns[1:-1],outputCol='features')
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

'''
# 模型预测
predict_set = sql_context.sql('select * from test.gold_medal_stock')
print('predict_set count: %d' % predict_set.count())
predict_set = pipeline_model.transform(predict_set)
print('predict_set count: %d' % predict_set.count())
predict_set.show(1)

# 预测结果保存到Hive
sql_context.sql('DROP TABLE IF EXISTS test.gold_medal_stock_prediction')
predict_set.select('cust_pty_no', 'prob', 'prediction').registerTempTable('gold_medal_stock_prediction')
sql = 'CREATE TABLE test.gold_medal_stock_prediction AS SELECT * FROM gold_medal_stock_prediction'
sql_context.sql(sql)
print('save prediction to hive done')

'''