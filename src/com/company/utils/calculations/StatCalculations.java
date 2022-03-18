package com.company.utils.calculations;

import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.company.utils.calculations.MathCalculations.*;
import static java.lang.Math.abs;
import static jsat.linear.distancemetrics.PearsonDistance.correlation;

public class StatCalculations {

	static public boolean isStrictlyIncreasing (SimpleDataSet dataSet, int column) {
		double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
		for (int i = 1; i < dataSet.getSampleSize(); i++) {
			if (Double.compare(dataSet.getDataPoint(i).getNumericalValues().get(column), current) <= 0) {
				return false;
			}
			current = dataSet.getDataPoint(i).getNumericalValues().get(column);
		}
		return true;
	}

	static public boolean isStrictlyDecreasing (SimpleDataSet dataSet, int column) {
		double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
		for (int i = 1; i < dataSet.getSampleSize(); i++) {

			if (Double.compare(dataSet.getDataPoint(i).getNumericalValues().get(column), current) >= 0) {
				return false;
			}
			current = dataSet.getDataPoint(i).getNumericalValues().get(column);
		}
		return true;
	}

	static public boolean isCloseToMean (SimpleDataSet dataSet, Statistics statistics) {
		double std = dataSet.getDataMatrix().getColumn(statistics.columnPredicted).standardDeviation();
		double mean = dataSet.getDataMatrix().getColumn(statistics.columnPredicted).mean();
		return std / mean <= statistics.getThresholds()[0];
	}

	static public boolean isCloseToMedian (SimpleDataSet dataSet, Statistics statistics) {
		double dist = 0.0;
		double median = dataSet.getDataMatrix().getColumn(statistics.columnPredicted).median();
		for (DataPoint dp : dataSet.getDataPoints()) {
			dist += abs(dp.getNumericalValues().get(statistics.columnPredicted) - median);
		}
		return dist / 8 / median <= statistics.getThresholds()[1];
	}

	static public boolean hasLinearRelationship (SimpleDataSet dataSet, Statistics statistics) {
		if (statistics.columnPredictors.length > 1) {
			//calculate pearson correlation with one predictor
			return !(getCorrMultiple(dataSet, statistics.columnPredicted, statistics.columnPredictors) > statistics.getThresholds()[2]);
		} else {
			//calculate pearson correlation with multiple predictors
			double corr = correlation(dataSet.getNumericColumn(statistics.columnPredicted), dataSet.getNumericColumn(statistics.columnPredictors[0]), true);
			return !(abs(corr) < statistics.getThresholds()[2]);
		}
	}

	/**
	 * Get the most optimal polynomial order.
	 *
	 * @param dataSet
	 * @param statistics
	 */
	static public int getPolynomialOrder (SimpleDataSet dataSet, Statistics statistics) {
		int n = dataSet.getSampleSize();
		//create raw polynomials of values in columnPredictor
		double[][] powPred = new double[4][n];
		for (int pow = 0; pow < 4; pow++) {
			for (int index = 0; index < n; index++) {
				double x = dataSet.getDataPoint(index).getNumericalValues().get(statistics.columnPredictors[0]);
				powPred[pow][index] = getPolyWithoutCoefficients(x, pow);
			}
		}
		// calculate correlations between raw polynomials and values in columnPredicted
		double[] corr = new double[4];
		for (int pow = 1; pow < 4; pow++) {
			corr[pow] = correlation(dataSet.getNumericColumn(statistics.columnPredicted), new DenseVector(powPred[pow]), true);
		}
		int iMax = getMax(corr); //choose the best correlation
		if (abs(corr[iMax]) < statistics.getThresholds()[3]) {
			return -1;
		}
		return iMax + 1;
	}

	/**
	 * Calculate pearson correlation with multiple predictors
	 *
	 * @param dataSet dataset with missing values
	 * @param columnPredicted index of the predicted column
	 * @param columnPredictor array of indexes of columns used for predicting
	 * @return correlation for multiple
	 */
	public static double getCorrMultiple (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictor) {
		int col = columnPredictor.length + 1;

		//count c^T
		RealMatrix mat = new BlockRealMatrix(dataSet.getSampleSize(), columnPredictor.length + 1);
		for (int i = 0; i < col - 1; i++) {
			mat.setColumn(i, dataSet.getDataMatrix().getColumn(columnPredictor[i]).arrayCopy());
		}
		mat.setColumn(col - 1, dataSet.getDataMatrix().getColumn(columnPredicted).arrayCopy());
		PearsonsCorrelation corr = new PearsonsCorrelation(mat);
		RealMatrix vector = corr.getCorrelationMatrix().getSubMatrix(col - 1, col - 1, 0, corr.getCorrelationMatrix().getColumnDimension() - 2);

		// count Rxx
		RealMatrix mat1 = new BlockRealMatrix(dataSet.getSampleSize(), columnPredictor.length);
		for (int i = 0; i < columnPredictor.length; i++) {
			mat1.setColumn(i, dataSet.getDataMatrix().getColumn(columnPredictor[i]).arrayCopy());
		}
		PearsonsCorrelation corr1 = new PearsonsCorrelation(mat1);
		RealMatrix rxx = corr1.getCorrelationMatrix().scalarMultiply(1 / new LUDecomposition(corr1.getCorrelationMatrix()).getDeterminant());

		RealMatrix a = vector.multiply(vector.transpose());
		RealMatrix b = vector.multiply(rxx).multiply(vector.transpose());

		return a.getEntry(0, 0) / b.getEntry(0, 0);
	}

	/**
	 * Calculate the statistics of the predicted column/-s
	 *
	 * @param columnPredicted predicted
	 * @param columnPredictors predictors
	 * @param datasetMissing incomplete dataset
	 */
	public static Map<Integer, Statistics> calcStatistics (int columnPredicted, int[] columnPredictors, SimpleDataSet datasetMissing) {
		Map<Integer, Statistics> statistics = new HashMap<>();
		if (columnPredicted != -1) {
			Statistics statistic = columnPredictors.length == 1
				? new Statistics(datasetMissing, columnPredicted, columnPredictors[0])
				: new Statistics(datasetMissing, columnPredicted, columnPredictors);
			statistics.put(columnPredicted, statistic);
		} else {
			int n = datasetMissing.getNumericColumns().length;
			int[] arrColumns = new int[n];
			for (int i = 0; i < n; i++) {
				arrColumns[i] = i;
			}
			int[] predicted = getDifference(arrColumns, columnPredictors);
			for (int j : predicted) {
				Statistics statistic = columnPredictors.length == 1
					? new Statistics(datasetMissing, j, columnPredictors[0])
					: new Statistics(datasetMissing, j, columnPredictors);
				statistics.put(j, statistic);
			}
		}
		return statistics;
	}

	/**
	 * Checks if all values in array are the same
	 * @param arr
	 */
	public static boolean allEqual (double[] arr) {
		return Arrays.stream(arr).distinct().count() == 1;
	}

	/**
	 * Checks whether the value is within the max and min of the column
	 * @param val
	 * @param statistics object that contain min and max
	 */
	public static boolean isWithinMaxAndMin (double val, Statistics statistics) {
		return Double.compare(statistics.getPercentiles()[0], val) <= 0 &&
			Double.compare(statistics.getPercentiles()[8], val) >= 0;
	}
}