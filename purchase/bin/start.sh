#!/usr/bin/env bash

PROJECT_PATH=$(cd "$(dirname "$0")"/..; pwd)
SPARK_HOME=/opt/software/spark-1.6.0-SNAPSHOT-bin-2.2.0

cd ${PROJECT_PATH}

${SPARK_HOME}/bin/spark-submit \
  --master yarn \
  --name Repurchase \
  --driver-memory 8g \
  --num-executors 8 \
  --executor-cores 6 \
  --executor-memory 16g \
  --conf "spark.driver.extraJavaOptions=-Xss1024m" \
  --conf "spark.executor.extraJavaOptions=-Xss1024m" \
  --conf "spark.sql.catalogImplementation=hive" \
  ${PROJECT_PATH}/src/re_purchase.py $*
