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
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import mean
from pyspark.sql.functions import rand

class PySparkMissForest:
       """
    Parameters
    ----------
    classifier : estimator object. the default is Random Forest Classifier.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.

    regressor : estimator object. The default is Random Forest Regressor.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.

     n_iter : int
     Determines the number of iteration.

     initial_guess : string, callable or None, default='median'
     If ``mean``, the initial impuatation will use the median of the features.
     If ``median``, the initial impuatation will use the median of the features.
    """
    
    def __init__(self, classifier=None, regressor=None, initial_guess='mean', n_iter=10,
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
            if len(binary_int_columns) == 0 or binary_int_columns is None:
                raise RuntimeError("""if convert_binary_int set True 
                               then binary_int_columns should be at least len 1 list""")
            for col_name in binary_int_columns:
                df = df.withColumn(col_name, df[col_name].cast("string"))
                
        # Initialize list to store names of columns converted from binary float to integer
        binary_float_to_int_cols = []

       # Check each float column to see if it's binary, but first sample a subset
        if convert_binary_float:
            sample_size = 200  # Adjust this based on the data size and distribution
            for field in df.schema.fields:
                if field.dataType == FloatType():
                    col_name = field.name
                    
                    # Sample the DataFrame
                    sample_df = df.orderBy(rand()).limit(sample_size)
                    
                    # Check distinct values in the sample
                    distinct_sample_values = sample_df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
                    
                    # If the sample suggests it's binary, proceed with the full check
                    if set(distinct_sample_values).issubset({0.0, 1.0}):
                        distinct_values = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
                        
                        if set(distinct_values) == {1.0, 0.0}:
                            df = df.withColumn(col_name, df[col_name].cast(LongType()))
                            binary_float_to_int_cols.append(col_name)

                    
        # Add an index column to the DataFrame
        df = df.withColumn("row_index", monotonically_increasing_id())

        # Get the total number of rows
        total_rows = df.count()

        # Get summary statistics
        summary = df.describe().filter("summary = 'count'")

        # Convert the row to a dictionary {column_name: count}
        count_row = {col: int(row[col]) for col, row in zip(summary.schema.names[1:], summary.collect())}

        # Identify columns with missing values
        miss_col = [col for col, count in count_row.items() if count < total_rows]
        
        # Sample function to get missing rows for a given column
        def get_missing_rows(df, col_name):
            return df.filter(df[col_name].isNull()).withColumn("missing_col", lit(col_name))
        # Union all the DataFrames
        miss_row_df = None
        for c in miss_col:
            temp_df = get_missing_rows(df, c)
            if miss_row_df is None:
                miss_row_df = temp_df
            else:
                miss_row_df = miss_row_df.union(temp_df)

        # Find rows with no missing values
        obs_row_df = df.dropna().select("row_index")
        
        #########

        # Create mappings for string columns
        mappings = {}
        rev_mappings = {}

        for field in df.schema.fields:
            col_type = field.dataType
            col_name = field.name
            if col_type == StringType():
                # Potentially expensive if many unique values
                unique_vals = df.agg(F.collect_set("column_name")).first()[0]
                mappings[col_name] = {k: v for k, v in zip(unique_vals, range(len(unique_vals)))}
                rev_mappings[col_name] = {v: k for k, v in zip(unique_vals, range(len(unique_vals)))}

        # Find columns that won't be imputed
        non_impute_cols = [c for c in df.columns if c not in mappings.keys()]
        
        ###
        # Precompute mean values for numeric columns (if needed)
        if self.initial_guess == 'mean':
            mean_vals = df.select([mean(c).alias(c) for c in df.columns if isinstance(df.schema[c].dataType, (IntegerType, FloatType))]).collect()[0].asDict()

        # Categorize columns
        numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, FloatType))]
        long_cols = [field.name for field in df.schema.fields if field.dataType == LongType()]
        categorical_cols = [field.name for field in df.schema.fields if field.dataType == StringType()]

        # Fill missing values for numeric columns
        if self.initial_guess == 'mean':
            df = df.na.fill(mean_vals)
        else:
            print("Median and mode computation in PySpark is expensive and currently not supported.")
            print('Initial guess is set to mean')
            df = df.na.fill(mean_vals)

        # Fill missing values for LongType columns
        if self.convert_binary_float:
            mode_vals_long = df.groupby(long_cols).count().orderBy('count', ascending=False).first()
            fill_values_long = {col: mode_vals_long[i] for i, col in enumerate(long_cols)}
            df = df.na.fill(fill_values_long)

        # Fill missing values for categorical columns
        mode_vals_cat = {col: df.groupby(col).count().orderBy('count', ascending=False).first()[0] for col in categorical_cols}
        df = df.na.fill(mode_vals_cat)
        
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
                
                ######
                # Skip if there's no missing value
                missing_rows_for_c = miss_row_df.\
                    filter(miss_row_df["missing_col"] == c).select("row_index")
                if missing_rows_for_c.count() == 0:
                    continue
                # Use only the columns that will be imputed for creating the feature vector
                assembler = VectorAssembler(inputCols=[col for col in impute_cols if col != c], outputCol="features")

        
                # Decide the estimator based on column type
                if c in mappings or c in binary_float_to_int_cols:
                    estimator = classifier
                else:
                    estimator = regressor
                
                # Narrow down DataFrame to only include relevant columns for imputation plus any auxiliary columns
                narrowed_df = df.select(impute_cols + ["row_index"])

                # Prepare data for training
                obs_data = narrowed_df.join(obs_row_df, on="row_index", how="inner").filter(narrowed_df[c].isNotNull())
                obs_data = assembler.transform(obs_data)
                obs_data = obs_data.select(["features", c])


                # Fit estimator
                model = estimator.fit(obs_data)
                
                # Prepare data for prediction
                miss_data = df.join(missing_rows_for_c, on="row_index", how="inner")
                miss_data = assembler.transform(miss_data)

                # Make predictions
                predictions = model.transform(miss_data)
                
                ######
                
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
            ###
            # Only select the columns that were originally part of `miss_col`
            relevant_cols = [c for c in df.columns if c in miss_col]

            # Count missing values only for relevant columns
            missing_count_df = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in impute_cols]).collect()
            missing_count = missing_count_df[0].asDict()

            # Identify columns with missing values
            miss_col = [c for c, count in missing_count.items() if count > 0]

            # Check termination criteria
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
        
        # Drop the index column
        df = df.drop("row_index")
        
        return df
