from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import FloatType, LongType  # Importing LongType for Int64
from pyspark.sql.functions import isnan, when, count
from pyspark.sql.functions import create_map, lit
from itertools import chain
from pyspark.sql.functions import round as spark_round
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame

class PySparkMissForest:
       """
    Parameters
    ----------
    classifier : estimator object.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.

    regressor : estimator object.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.

     n_iter : int
     Determines the number of iteration.

     initial_guess : string, callable or None, default='median'
     If ``mean``, the initial impuatation will use the median of the features.
     If ``median``, the initial impuatation will use the median of the features.
    """
    
    def __init__(self, classifier=None, regressor=None, initial_guess='median', n_iter=10,
                 convert_binary_int=True, binary_int_columns=None, convert_binary_float=False):
        self.classifier = classifier if classifier else RandomForestClassifier()
        self.regressor = regressor if regressor else RandomForestRegressor()
        self.initial_guess = initial_guess
        self.n_iter = n_iter
        self.convert_binary_int = convert_binary_int
        self.binary_int_columns = binary_int_columns
        self.convert_binary_float = convert_binary_float
        
    def fit_transform(self, df: DataFrame) -> DataFrame:
        convert_binary_int = self.convert_binary_int
        binary_int_columns = self.binary_int_columns
        convert_binary_float = self.convert_binary_float
        
        
        
        if convert_binary_int:
             if len(binary_int_columns) == 0:
                raise RuntimeError("""if convert_binary_int set True 
                               then binary_int_columns should be at least len 1 list""")
            for col_name in binary_int_columns:
                df = df.withColumn(col_name, df[col_name].cast("string"))

        # Convert float columns that are binary to Int64 (LongType in PySpark)
        if convert_binary_float:
            for field in df.schema.fields:
                if field.dataType == FloatType():
                    col_name = field.name
                    df = df.withColumn(col_name, df[col_name].cast(LongType()))
                    
       # Add an index column to the DataFrame
        df = df.withColumn("row_index", monotonically_increasing_id())

        # Initialize a dictionary to keep track of missing rows for each column
        miss_row = {}

        # Identifying columns with missing values
        miss_col = [c for c in df.columns if df.where(df[c].isNull()).count() > 0]
        
        # Record the row indices for missing values in each column
        for c in miss_col:
            miss_row[c] = df.filter(df[c].isNull()).select("row_index").rdd.flatMap(lambda x: x).collect()

        
        # Find rows with no missing values
        obs_row = df.dropna().select("row_index").rdd.flatMap(lambda x: x).collect()

        # Create mappings for string columns
        mappings = {}
        rev_mappings = {}
        for field in df.schema.fields:
            col_type = field.dataType
            col_name = field.name
            if col_type == StringType():
                unique_vals = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
                mappings[col_name] = {k: v for k, v in zip(unique_vals, range(len(unique_vals)))}
                rev_mappings[col_name] = {v: k for k, v in zip(unique_vals, range(len(unique_vals)))}

        # Find columns that won't be imputed
        non_impute_cols = [c for c in df.columns if c not in mappings.keys()]
        
        for field in df.schema.fields:
            col_type = field.dataType
            col_name = field.name

            # If datatype is numeric, fillna with mean or median
            if isinstance(col_type, (IntegerType, FloatType)):
                if self.initial_guess == 'mean':
                    mean_val = df.select(mean(col_name)).collect()[0][0]
                    df = df.na.fill({col_name: mean_val})
                else:
                    # Median computation in PySpark can be expensive, alternative logic may be used here
                    pass

            # If datatype is Int64 and convert_binary_float is True
            elif col_type == LongType() and self.convert_binary_float:
                # Assuming mode computation logic here
                mode_val = df.groupby(col_name).count().orderBy('count', ascending=False).select(col_name).limit(1).collect()[0][0]
                df = df.withColumn(col_name, when(col(col_name).isNull(), mode_val).otherwise(col(col_name)).cast(LongType()))

            # If datatype is categorical, fillna with mode
            else:
                mode_val = df.groupby(col_name).count().orderBy('count', ascending=False).select(col_name).limit(1).collect()[0][0]
                df = df.na.fill({col_name: mode_val})
         # Label Encoding
        for c, mapping in mappings.items():
            mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
            df = df.withColumn(c, mapping_expr[df[c]].cast(IntegerType()))
            
        # Initialize the estimator
        classifier = RandomForestClassifier()
        regressor = RandomForestRegressor()
        
        # Column Iteration and Estimator Selection:
        iter = 0
        while True:
            for c in miss_col:
                assembler = VectorAssembler(inputCols=[col for col in df.columns if col != c], outputCol="features")
                
                # Decide the estimator based on column type
                estimator = classifier if c in mappings else regressor
                # Prepare data for training
                obs_data = df.filter(df[c].isNotNull())
                obs_data = assembler.transform(obs_data)
                obs_data = obs_data.select(["features", c])

                # Fit estimator
                model = estimator.fit(obs_data)
                # Predict the missing column with the trained estimator
                # Prepare data for prediction
                miss_data = df.filter(df[c].isNull())
                miss_data = assembler.transform(miss_data)

                # Make predictions
                predictions = model.transform(miss_data)
                
                # Extract predictions and join them back to the original DataFrame
                predicted_values = predictions.select("row_index", "prediction")
                df = df.join(predicted_values, on="row_index", how="left_outer")

                # Update the original DataFrame based on data type
                col_type = [field.dataType for field in df.schema.fields if field.name == c][0]
                if isinstance(col_type, LongType):  # Equivalent to 'int64' in pandas
                    df = df.withColumn(c, when(df[c].isNull(), spark_round(df["prediction"])).otherwise(df[c]))
                else:
                    df = df.withColumn(c, when(df[c].isNull(), df["prediction"]).otherwise(df[c]))

                # Drop the 'prediction' column as it's no longer needed
                df = df.drop("prediction")
                
                # Check if criteria is met
            if iter >= self.n_iter:
                break

            iter += 1
            
            # Check if there are any columns left with missing values
            miss_col = [c for c in df.columns if df.where(df[c].isNull()).count() > 0]
            if len(miss_col) == 0:
                break
            else:
                iter += 1
                if iter > self.n_iter:
                    raise Exception("Maximum number of iterations reached")
                    
        # Reverse mapping
        for c, rev_mapping in rev_mappings.items():
            rev_mapping_expr = create_map([lit(x) for x in chain(*rev_mapping.items())])
            df = df.withColumn(c, rev_mapping_expr[df[c]])

        

        # Printing mappings for debugging or logging purposes
        print("mappings:", mappings)
        print("rev_mappings:", rev_mappings)
        
        # Drop the index column
        df = df.drop("row_index")
        self.df = df
        return df
            
        