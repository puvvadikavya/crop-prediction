from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
spark = SparkSession.builder.master(
    "local").appName("Crop Prediction").getOrCreate()
df = spark.read.csv("crop.csv", header=True, inferSchema=True)
# import PySpark libraries


# drop any null values
data = df.dropna()

# convert categorical variable "Crop" to numerical values
crop_indexer = StringIndexer(inputCol='Crop', outputCol='CropIndex')
data = crop_indexer.fit(data).transform(data)

# create feature vector from input columns
assembler = VectorAssembler(inputCols=[
                            'N', 'P', 'K', 'pH', 'EC', 'OC', 'B', 'Zn', 'Fe', 'Mn', 'Cu', 'S'], outputCol='features')
data = assembler.transform(data)

# standardize the input features
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# normalize the numerical columns
numerical_columns = ['N', 'P', 'K', 'pH', 'EC',
                     'OC', 'B', 'Zn', 'Fe', 'Mn', 'Cu', 'S']
for column in numerical_columns:
    min_val = data.agg({column: "min"}).collect()[0][0]
    max_val = data.agg({column: "max"}).collect()[0][0]
    data = data.withColumn(
        column, (col(column) - min_val) / (max_val - min_val))
# split the preprocessed data into training and testing datasets
(training_data, testing_data) = data.randomSplit([0.8, 0.2])

# create a RandomForestClassifier object
rf = RandomForestClassifier(
    featuresCol='scaled_features', labelCol='CropIndex', numTrees=10, maxDepth=5)

# fit the model to the training data
model = rf.fit(training_data)

# make predictions on the testing data
predictions = model.transform(testing_data)

# evaluate the performance of the model using the RMSE metric
evaluator = RegressionEvaluator(
    labelCol='CropIndex', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)

# print the RMSE metric
print('RMSE:', rmse)
# evaluate the performance of the model using the RMSE metric
evaluator = MulticlassClassificationEvaluator(
    labelCol='CropIndex', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)

# print the accuracy
print('Accuracy:', accuracy)


def prediction(N, P, K, pH, EC, OC, B, Zn, Fe, Mn, Cu, S):

    # Create a DataFrame with the new data to predict
    new_data = spark.createDataFrame([(N, P, K, pH, EC, OC, B, Zn, Fe, Mn, Cu, S)],
                                     ['N', 'P', 'K', 'pH', 'EC', 'OC', 'B', 'Zn', 'Fe', 'Mn', 'Cu', 'S'])

    # Apply pre-processing steps to the new data
    # (e.g. normalization using the same MinMaxScaler object used for the original dataset)

    # Use the VectorAssembler and MinMaxScaler objects to transform the new data
    new_data_transformed = assembler.transform(new_data)
    new_data_transformed = scaler_model.transform(new_data_transformed)

    # Select only the features used to train the model
    new_data_transformed = new_data_transformed.select('scaled_features')

    # Make predictions on the new data using the trained model
    predictions = model.transform(new_data_transformed)

    # Select the prediction column
    predicted_crop_index = predictions.select('prediction').collect()[0][0]

    from pyspark.sql.functions import collect_list

    # Get distinct CropIndex and Crop pairs
    crop_pairs_df = data.select('CropIndex', 'Crop').distinct()

    # Group the DataFrame by CropIndex and collect the corresponding Crop name
    crop_index_df = crop_pairs_df.groupBy('CropIndex').agg(
        collect_list('Crop').alias('CropList'))

    # Convert the DataFrame to a Python dictionary where CropIndex is the key and Crop is the value
    crop_index_dict = crop_index_df.rdd.collectAsMap()

    # extract the prediction value from the predictions DataFrame
    prediction_value = float(predictions.select('prediction').first()[0])

    # use the dictionary to get the crop name from the predicted CropIndex
    predicted_crop = crop_index_dict.get(int(prediction_value), 'Unknown')

    # return the predicted crop name
    return predicted_crop
