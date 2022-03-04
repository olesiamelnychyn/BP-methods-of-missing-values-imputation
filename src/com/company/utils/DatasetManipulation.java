package com.company.utils;

import com.company.ConfigManager;
import com.company.utils.calculations.StatCalculations;
import com.company.utils.objects.MainData;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.io.CSV;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static com.company.utils.calculations.MathCalculations.*;

public class DatasetManipulation {
	static ConfigManager configManager = ConfigManager.getInstance();

	/** Encode Missingness
	 *
	 * @param filename original filename
	 * @return filename of the dataset with missingness encoded
	 *
	 * This method replaces different representations of missing values in one which is appropriate for jsat.io.CSV.read(),
	 * namely - empty string
	 */
	private static String encodeMissingness (String filename) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		filename = filename.replace(".csv", "_NULL.csv");
		filename = filename.replace("data", "data/ignored");
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename));

		String out = br.lines()
			.map(l -> l.split(",", -1))
			.map(arr -> Arrays.stream(arr).map(value ->
				("null".equalsIgnoreCase(value) ||
					"?".equals(value) ||
					("na").equalsIgnoreCase(value) ||
					("nan").equalsIgnoreCase(value) ||
					value.length() == 0 ||
					(Boolean.parseBoolean(configManager.get("isZeroMissing")) && "0".equals(value))) ?
					"" :
					value
			).collect(Collectors.joining(",")))
			.collect(Collectors.joining("\n"));
		bw.write(out);
		br.close();
		bw.close();
		return filename;
	}

	/** Read Dataset
	 *
	 * @param fileName original filename
	 * @param containMissing if it contains missing values
	 * @return dataset read from file
	 *
	 * Reads dataset from file. In case it contains missing values, it firstly encodes missingnes,
	 * so that jsat.io.CSV.read() do not throw exception.
	 */
	static public SimpleDataSet readDataset (String fileName, boolean containMissing) throws IOException {
		if (containMissing) {
			fileName = encodeMissingness(fileName);
		}

		SimpleDataSet simpleDataSet = CSV.read(Paths.get(fileName), ',', 0, ' ', new HashSet<>());
		//count number of missing values
		int nans = 0;
		for (DataPoint dp : simpleDataSet.getDataPoints()) {
			if (dp.getNumericalValues().countNaNs() != 0) {
				nans++;
			}
		}
		System.out.println("Number of missing values: " + nans);
		return simpleDataSet;
	}

	static public SimpleDataSet createDeepCopy (SimpleDataSet dataset, int from, int to) {
		List<DataPoint> ll = new ArrayList<>();
		ll.add(dataset.getDataPoint(from++).clone());
		SimpleDataSet datasetCopy = new SimpleDataSet(ll);
		for (DataPoint obj : dataset.getDataPoints().subList(from, to)) {
			datasetCopy.add(obj.clone());
		}
		return datasetCopy;
	}

	/** Rank records in dataset by euclidean distance from the main record to be imputed
	 *
	 * @param dataset  original dataset
	 * @param mainDP record where null was found
	 * @param columnPredictors index(es) of column(s) to be used for prediction
	 * @param firstIndex where the search for training records should start
	 * @param index index of mainDp
	 * @param weighted whether to add weights to records
	 * @return ordered map of records by euclidean from mainDP
	 */
	static public TreeMap<Double, DataPoint> rankDatasetByEuclideanDistance (SimpleDataSet dataset, DataPoint mainDP, int[] columnPredictors, int firstIndex, int index, boolean weighted) {
		TreeMap<Double, DataPoint> rankedRecords = new TreeMap<>();
		ArrayList<DataPoint> records = new ArrayList<>();
		// count distance for records before current
		if (firstIndex < index) {
			for (DataPoint dp : dataset.getDataPoints().subList(firstIndex, index - 1)) {
				if (getIntersection(getIndexesOfNull(dp), columnPredictors).length == 0) { // only if predictors are not null
					records.add(dp);
				}
			}
		}

		// count distance for records after current
		int lastIndex = Math.min(dataset.getSampleSize(), index + 30);
		for (DataPoint dp : dataset.getDataPoints().subList(index + 1, lastIndex)) {
			if (getIntersection(getIndexesOfNull(dp), columnPredictors).length == 0) { // only if predictors are not null
				records.add(dp);
			}
		}

		SimpleDataSet recordsDS = new SimpleDataSet(records);
		recordsDS.add(mainDP);
		int k = 0;
		double[] maxAndMin = new double[columnPredictors.length * 2];
		for (int columnPredictor : columnPredictors) {
			Vec column = recordsDS.getDataMatrix().getColumn(columnPredictor);
			maxAndMin[k++] = column.max();
			maxAndMin[k++] = column.min();
		}

		records.remove(mainDP);
		// normalize predictors in mainDP
		double[] valuesMain = normilizePredictors(mainDP, columnPredictors, maxAndMin, recordsDS.getSampleSize());

		records.forEach((dp) -> {
			// normalize predictors in dp
			double[] valuesSec = normilizePredictors(dp, columnPredictors, maxAndMin, recordsDS.getSampleSize());
			double dist = getEuclideanDistance(valuesMain, valuesSec);
			if (weighted) {
				dp.setWeight(getWeightByEuclidean(dist));
			}

			rankedRecords.put(dist, dp);
		});
		return rankedRecords;
	}

	/** Split datasetMissing
	 *
	 * @param datasetMissing original datasetMissing
	 * @param index index of current record
	 * @param nRecords number of records in training dataset to be returned
	 * @param weighted whether to add weights to records
	 *
	 *  This method splits data records close to the predicted ones by euclidean distance into those which will be used for the prediction and
	 *  those which will be predicted (only those which are in close euclidean distance and have missing value belong here).
	 */
	static public void getToBeImputedAndTrainDeepCopiesByClosestDistance (MainData data, SimpleDataSet datasetMissing, int index, int nRecords, boolean weighted) {
		List<DataPoint> dataPointsTrain = new ArrayList<>(); //records used for training
		List<DataPoint> dataPointsToBeImputed = new ArrayList<>();//records in which values should be predicted
		DataPoint mainDP = datasetMissing.getDataPoints().get(index);
		int firstIndex = index - 30; //index of the first record, by default 4 records before current one
		int nTraining = 0; //number of records used for training

		dataPointsToBeImputed.add(mainDP); //current record is always the one to be predicted
		if (firstIndex != -30) { //if current record is not the first in datasetMissing
			if (firstIndex < 0) { //if current record is 2nd/3rd/etc.
				firstIndex = 0;
			}
		} else { //if current is first -> we start to search for training data from next
			firstIndex = index + 1;
		}

		TreeMap<Double, DataPoint> ranked = rankDatasetByEuclideanDistance(datasetMissing, mainDP, data.getColumnPredictors(), firstIndex, index, weighted);

		// split ranked records until training dataset contains nRecords
		for (DataPoint dp : ranked.values()) {
			if (!Double.isNaN(dp.getNumericalValues().get(data.getColumnPredicted()))) {
				dataPointsTrain.add(dp);
				nTraining++;
				if (nTraining >= nRecords) {
					break;
				}
			} else if (datasetMissing.countMissingValues() > 1000000) {
				// Creating toBePredicted dataset is only for time-consuming boost,
				// but at the same time it can worsen results, therefore the condition is so big
				dataPointsToBeImputed.add(dp);
			}

		}
		if (weighted) {
			normalizeWeights(dataPointsTrain);
		}

		data.setImpute(new SimpleDataSet(dataPointsToBeImputed));
		data.setTrain(new SimpleDataSet(dataPointsTrain));
		data.setColumnPredictors(Arrays.stream(data.getColumnPredictors())
			.filter(pred -> !StatCalculations.allEqual(data.getTrain().getDataMatrix().getColumn(pred).arrayCopy()))
			.toArray());
	}

	/** Split dataset
	 *
	 * @param dataset original dataset
	 * @param index index of current record
	 * @param nRecords number of records in training dataset to be returned
	 *
	 * This method splits data records close to the predicted index into those which will be used for the prediction and
	 * those which will be predicted (since they are close to each other, there might be no sense in predicting them separately)
	 */
	static public void getToBeImputedAndTrainDeepCopiesAroundIndex (MainData data, SimpleDataSet dataset, int index, int nRecords) {
		List<DataPoint> dataPointsTrain = new ArrayList<>(); //records used for training
		List<DataPoint> dataPointsToBeImputed = new ArrayList<>();//records in which values should be predicted
		int firstIndex = index - nRecords / 2; //index of the first record, by default 4 records before current one
		int nTraining = 0; //number of records used for training

		dataPointsToBeImputed.add(dataset.getDataPoints().get(index)); //current record is always the one to be predicted
		if (firstIndex != -nRecords / 2) { //if current record is not the first in dataset
			if (firstIndex < 0) { //if current record is 2nd/3rd
				firstIndex = 0;
			}
			//records before current are always assumed to be already complete as we traverse the dataset from beginning
			for (DataPoint obj : dataset.getDataPoints().subList(firstIndex, index)) {
				dataPointsTrain.add(obj.clone());
				nTraining++;
			}
		}

		int n = dataset.getSampleSize(); //the last index in dataset
		//filling dataset used for prediction till we have nRecords in it or there are no more records in dataset after current
		for (int i = index + 1; i < n && nTraining < nRecords; i++) {
			DataPoint obj = dataset.getDataPoint(i);

			//if a record has one or more of the predictors' equal to null it cannot be used for prediction
			if (getIntersection(getIndexesOfNull(obj), data.getColumnPredictors()).length == 0) {
				//if record contains value in column which is going to be predicted
				//then add it to the dataset used for prediction,
				if (!Double.isNaN(obj.getNumericalValues().get(data.getColumnPredicted()))) {
					dataPointsTrain.add(obj.clone());
					nTraining++;
				} else {
					//otherwise, add it to the dataset which is going to be predicted
					dataPointsToBeImputed.add(obj);
				}
			}
		}

		data.setImpute(new SimpleDataSet(dataPointsToBeImputed));
		data.setTrain(new SimpleDataSet(dataPointsTrain));
	}

	/** Convert columns of dataset to double[][] array
	 *
	 * @param dataset original dataset
	 * @param columns array of indexes of columns to be present in array
	 * @return reternes converted part of the dataset
	 *
	 * It converts dataset to an array, which contains only values in the specified columns and
	 * additional column as constant that equals to 1
	 */
	static public double[][] toArray (SimpleDataSet dataset, int[] columns) {
		double[][] array = new double[dataset.getSampleSize()][columns.length + 1];
		int i = 0;
		for (DataPoint obj : dataset.getDataPoints()) {
			int k = 0;
			array[i][k++] = 1;
			for (int j : columns) {
				array[i][k++] = obj.getNumericalValues().get(j);
			}
			i++;
		}
		return array;
	}

	/** Reverse dataset
	 *
	 * @param dataSet original dataset
	 * @return new reversed dataset
	 */
	static public SimpleDataSet reverseDataset (SimpleDataSet dataSet) {
		ArrayList<DataPoint> list = new ArrayList<>();
		for (int i = dataSet.getSampleSize() - 1; i >= 0; i--) {
			list.add(dataSet.getDataPoint(i));
		}
		return new SimpleDataSet(list);
	}

	/** Remove columns of non-predictors
	 *
	 * @param dataset original dataset
	 * @param degree max degree which should be in new dataset used for multiple polynomial regression
	 * @param columnPredicted index of column to be predicted
	 * @param predictors index(es) of column(s) to be used for prediction
	 * @return new dataset which contains columns with powers of values (from 1 to degree)
	 *
	 * This method add columns of predictors' powers to the dataset
	 */
	public static SimpleDataSet addPowerColumns (SimpleDataSet dataset, int degree, int[] predictors, int columnPredicted) {
		List<DataPoint> list = new ArrayList<>();
		for (DataPoint dp : dataset.getDataPoints()) {
			// new numerical values of a record which look like: [x1 x2... x1^2 x2^2... ... x1^n x2^n...]
			Vec vec = new DenseVector(predictors.length + 1);
			vec.set(columnPredicted, dp.getNumericalValues().get(columnPredicted));
			int next = predictors.length / degree + 1;
			for (int i = 0; i < dataset.getDataMatrix().cols(); i++) {
				if (i != columnPredicted) {
					vec.set(i, dp.getNumericalValues().get(i)); //set x
					for (int j = 2; j <= degree; j++) {
						vec.set(next, Math.pow(dp.getNumericalValues().get(i), j)); //set x^2, ..., x^n
						next++;
					}
				}
			}

			list.add(new DataPoint(vec, dp.getWeight()));
		}
		return new SimpleDataSet(list);
	}

	/** Remove columns of non-predictors
	 *
	 * @param dataset original dataset
	 * @param columnPredicted index of column to be predicted
	 * @param predictors index(es) of column(s) to be used for prediction
	 * @return filename of the dataset with missingness encoded
	 *
	 * This method removes column(s) which is(are) not used for prediction
	 */
	public static SimpleDataSet excludeNonPredictors (SimpleDataSet dataset, int[] predictors, int columnPredicted) {
		List<DataPoint> list = new ArrayList<>();
		for (DataPoint dp : dataset.getDataPoints()) {
			//new data record which contains only predicted value and values of predictors
			Vec vec = new DenseVector(predictors.length + 1);
			vec.set(0, dp.getNumericalValues().get(columnPredicted));
			int nextIndex = 1;
			for (int column : predictors) {
				vec.set(nextIndex++, dp.getNumericalValues().get(column));
			}
			list.add(new DataPoint(vec, dp.getWeight()));
		}
		return new SimpleDataSet(list);
	}

	static public void printDataset (SimpleDataSet dataSet) {
		for (DataPoint dp : dataSet.getDataPoints()) {
			System.out.println(dp.getNumericalValues().toString() + " * " + dp.getWeight());
		}
	}

	static public void printDatasetColumn (SimpleDataSet dataSet, int column) {
		for (DataPoint dp : dataSet.getDataPoints()) {
			System.out.println(dp.getNumericalValues().get(column));
		}
	}

}
