# Databricks notebook source
df_actor = spark.read.csv("/Volumes/film/test_schema/pagila_data/actor.csv", header=True, inferSchema=True)
df_address = spark.read.csv("/Volumes/film/test_schema/pagila_data/address.csv", header=True, inferSchema=True)
df_category = spark.read.csv("/Volumes/film/test_schema/pagila_data/category.csv", header=True, inferSchema=True)
df_city = spark.read.csv("/Volumes/film/test_schema/pagila_data/city.csv", header=True, inferSchema=True)
df_country = spark.read.csv("/Volumes/film/test_schema/pagila_data/country.csv", header=True, inferSchema=True)
df_customer = spark.read.csv("/Volumes/film/test_schema/pagila_data/customer.csv", header=True, inferSchema=True)
df_film = spark.read.csv("/Volumes/film/test_schema/pagila_data/film.csv", header=True, inferSchema=True)
df_film_actor = spark.read.csv("/Volumes/film/test_schema/pagila_data/film_actor.csv", header=True, inferSchema=True)
df_film_category = spark.read.csv("/Volumes/film/test_schema/pagila_data/film_category.csv", header=True, inferSchema=True)
df_inventory = spark.read.csv("/Volumes/film/test_schema/pagila_data/inventory.csv", header=True, inferSchema=True)
df_language = spark.read.csv("/Volumes/film/test_schema/pagila_data/language.csv", header=True, inferSchema=True)
df_payment = spark.read.csv("/Volumes/film/test_schema/pagila_data/payment.csv", header=True, inferSchema=True)
df_rental = spark.read.csv("/Volumes/film/test_schema/pagila_data/rental.csv", header=True, inferSchema=True)
df_staff = spark.read.csv("/Volumes/film/test_schema/pagila_data/staff.csv", header=True, inferSchema=True)
df_store = spark.read.csv("/Volumes/film/test_schema/pagila_data/store.csv", header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 1) Выведите количество фильмов в каждой категории, отсортированных в порядке убывания. 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import count
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window

category_counts = df_film_category.groupBy("category_id").agg(count("*").alias("count"))
sorted_category_counts = category_counts.orderBy("count", ascending=False)
sorted_category_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2) Выведите 10 актеров, фильмы которых имели наибольший прокат, отсортированных в порядке убывания. 

# COMMAND ----------


result_df = (
    df_actor.alias("a")
    .join(df_film_actor.alias("fa"), F.col("a.actor_id") == F.col("fa.actor_id"))
    .join(df_inventory.alias("i"), F.col("fa.film_id") == F.col("i.film_id"))
    .join(df_rental.alias("r"), F.col("i.inventory_id") == F.col("r.inventory_id"))
    .groupBy("a.actor_id", "a.first_name", "a.last_name")
    .agg(F.count("r.rental_id").alias("total_rentals"))
    .orderBy(F.col("total_rentals").desc())
    .limit(10)
)
result_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 3) Выведите категорию фильмов, на которые было потрачено больше всего денег. 

# COMMAND ----------


most_money_df = (
    df_category.alias("c")
    .join(df_film_category.alias("fc"), col("c.category_id") == col("fc.category_id"), "left")
    .join(df_inventory.alias("inv"), col("fc.film_id") == col("inv.film_id"), "left")
    .join(df_rental.alias("r"), col("inv.inventory_id") == col("r.inventory_id"), "left")
    .join(df_payment.alias("p"), col("r.rental_id") == col("p.rental_id"), "left")
    .groupBy("c.name")
    .agg(F.sum("p.amount").alias("total_sum"))
)

window_spec = Window.orderBy(col("total_sum").desc())
result_df = (most_money_df.withColumn("RANK", F.dense_rank().over(window_spec)).filter(col("RANK") == 1))

result_df.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 4) Вывести названия фильмов, которых нет в каталоге. 

# COMMAND ----------


result_df = df_film.alias("f").join(df_inventory.alias("inv"), col("f.film_id") == col("inv.film_id"), "left_anti").select(col("f.film_id"), col("f.title"))

result_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 5) Выведите список из 3 актеров, которые чаще всего снимались в фильмах категории «Детские». Если у нескольких актеров одинаковое количество фильмов, выведите список всех актеров. 

# COMMAND ----------


most_actor_df = (
    df_actor.alias("a")
    .join(df_film_actor.alias("fa"),col("a.actor_id") == col("fa.actor_id"), "left")
    .join(df_film_category.alias("fc"), col("fa.film_id") == col("fc.film_id"), "left")
    .join(df_category.alias("c"), col("fc.category_id") == col("c.category_id"), "left")
    .filter(col("c.name") == 'Children')
    .groupBy("a.first_name","a.last_name")
    .agg(F.count("fa.film_id").alias("FILMUS_IN_CHILDREN"))
   
)

window_spec = Window.orderBy(col("FILMUS_IN_CHILDREN").desc())
result_df = (most_actor_df.withColumn("RANK", F.dense_rank().over(window_spec)).filter(col("RANK") <= 3))

result_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 6) Выведите список городов с указанием количества активных и неактивных клиентов (активные - customer.active = 1). Отсортируйте по количеству неактивных клиентов в порядке убывания. 

# COMMAND ----------



result_df = (
  df_city.alias("c")
  .join(df_address.alias("a"), col("c.city_id") == col("a.city_id"))
  .join(df_customer.alias("cu"), col("a.address_id") == col("cu.address_id"))
  .groupBy("c.city_id", "c.city") \
  .agg(F.count(F.when(col("cu.activebool") == True, 1)).alias("active_customers"),
       F.count(F.when(col("cu.activebool") == False, 1)).alias("inactive_customers"))
  .orderBy(col("inactive_customers").desc())
)
result_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 7) Выведите категорию фильмов, имеющих наибольшее общее количество часов проката в городах (customer.address_id в этом городе) и начинающихся с буквы «а». Сделайте то же самое для городов, отмеченных символом «-».

# COMMAND ----------


most_rental_df = (
    df_film.alias("f")
    .join(df_film_category.alias("fc"), col("f.film_id") == col("fc.film_id"), "left")
    .join(df_category.alias("c"), col("fc.category_id") == col("c.category_id"), "left")
    .join(df_inventory.alias("inv"), col("f.film_id") == col("inv.film_id"), "left")
    .join(df_rental.alias("r"), col("inv.inventory_id") == col("r.inventory_id"), "left")
    .join(df_customer.alias("cu"), col("r.customer_id") == col("cu.customer_id"), "left")
    .join(df_address.alias("add"), col("cu.address_id") == col("add.address_id"), "left")
    .join(df_city.alias("ci"), col("add.city_id") == col("ci.city_id"), "left")
    .filter((F.col("ci.city").like("a%")) | (F.col("ci.city").like("%-%")))
    .groupBy("c.name", "cu.address_id", "ci.city")
    .agg(F.sum("f.rental_duration").alias("total_rental"))
)
window_spec = Window.orderBy(F.col("total_rental").desc())
most_rental_ranked = most_rental_df.withColumn("RANK", F.dense_rank().over(window_spec))

result = most_rental_ranked.filter(F.col("RANK") == 1)
result.show()