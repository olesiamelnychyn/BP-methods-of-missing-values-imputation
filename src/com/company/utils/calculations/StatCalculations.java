package com.company.utils.calculations;

import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;

import java.util.*;

import static com.company.utils.calculations.MathCalculations.*;
import static java.lang.Math.*;
import static jsat.linear.distancemetrics.PearsonDistance.correlation;

public class StatCalculations {

	static public boolean isStrictlyIncreasing (MainData data, int column) {
		SimpleDataSet dataSet = data.getTrain();
		double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
		for (int i = 1; i < dataSet.getSampleSize(); i++) {
			if (Double.compare(dataSet.getDataPoint(i).getNumericalValues().get(column), current) <= 0) {
				return false;
			}
			current = dataSet.getDataPoint(i).getNumericalValues().get(column);
		}
		return true;
	}

	static public boolean isStrictlyDecreasing (MainData data, int column) {
		SimpleDataSet dataSet = data.getTrain();
		double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
		for (int i = 1; i < dataSet.getSampleSize(); i++) {

			if (Double.compare(dataSet.getDataPoint(i).getNumericalValues().get(column), current) >= 0) {
				return false;
			}
			current = dataSet.getDataPoint(i).getNumericalValues().get(column);
		}
		return true;
	}

	static public boolean isCloseToMean (MainData data, Statistics statistics) {
		SimpleDataSet dataSet = data.getTrain();
		double std = dataSet.getDataMatrix().getColumn(data.getColumnPredicted()).standardDeviation();
		double mean = dataSet.getDataMatrix().getColumn(data.getColumnPredicted()).mean();
		return std / mean <= statistics.getThresholds()[0];
	}

	static public boolean isCloseToMedian (MainData data, Statistics statistics) {
		SimpleDataSet dataSet = data.getTrain();
		double dist = 0.0;
		double median = dataSet.getDataMatrix().getColumn(data.getColumnPredicted()).median();
		for (DataPoint dp : dataSet.getDataPoints()) {
			dist += abs(dp.getNumericalValues().get(data.getColumnPredicted()) - median);
		}
		return dist / 8 / median <= statistics.getThresholds()[1];
	}

	static public boolean hasLinearRelationship (MainData data, Statistics statistics) {
		double corr;
		SimpleDataSet dataSet = data.getTrain();
		if (data.getColumnPredictors().length > 1) {
			//calculate pearson correlation with more than one predictor
			corr = getCorrMultiple(dataSet, data.getColumnPredicted(), data.getColumnPredictors());
		} else {
			//calculate pearson correlation with one predictor
			corr = correlation(dataSet.getNumericColumn(data.getColumnPredicted()), dataSet.getNumericColumn(data.getColumnPredictors()[0]), true);
		}
		return abs(corr) > statistics.getThresholds()[2];

	}

	/**
	 * Get the most optimal polynomial order.
	 * @param data dataset, predicted column and predictors together
	 * @param statistics statistics
	 * @return order
	 */
	static public int getPolynomialOrderSimple (MainData data, Statistics statistics) {
		SimpleDataSet dataSet = data.getTrain();
		int n = dataSet.getSampleSize();
		// create raw polynomials of values in columnPredictor
		// max order is 4, each row in powPred - has values in one order
		double[][] powPred = new double[4][n];
		for (int pow = 0; pow < 4; pow++) {
			for (int index = 0; index < n; index++) {
				double x = dataSet.getDataPoint(index).getNumericalValues().get(data.getColumnPredictors()[0]);
				powPred[pow][index] = getPolyWithoutCoefficients(x, pow);
			}
		}
		// calculate correlations between raw polynomials and values in columnPredicted
		double[] corr = new double[4];
		for (int pow = 1; pow < 4; pow++) {
			corr[pow] = correlation(dataSet.getNumericColumn(data.getColumnPredicted()), new DenseVector(powPred[pow]), true);
		}
		int iMax = getMax(corr); //choose the best correlation
		if (abs(corr[iMax]) < statistics.getThresholds()[3]) {
			return -1;
		}
		return iMax + 1;
	}

	/**
	 * Whether there is any polynomial relationship for multiple regression.
	 *
	 * @param data dataset, preicted column and predictors together
	 * @param statistics statistics
	 */
	static public boolean hasPolynomialRelation (MainData data, Statistics statistics) {
		SimpleDataSet dataSet = data.getTrain();
		ArrayList<DataPoint> dps = new ArrayList<>();

		dataSet.getDataPoints()
			.stream()
			.map(dp -> {
				DataPoint dataPoint = dp.clone();
				for (int pred : data.getColumnPredictors()) {
					dataPoint.getNumericalValues().set(pred, pow(dp.getNumericalValues().get(pred), 2));
				}
				return dataPoint;
			})
			.forEach(dps::add);
		SimpleDataSet rawPolySample = new SimpleDataSet(dps);

		// calculate correlations between raw polynomials and values in columnPredicted
		return abs(getCorrMultiple(rawPolySample, data.getColumnPredicted(), data.getColumnPredictors())) > statistics.getThresholds()[2];
	}

	/** Calculate pearson correlation with multiple predictors
	 *
	 * @param dataSet dataset
	 * @param columnPredicted predicted column
	 * @param columnPredictor predictors
	 * @return coefficient of multiple correlation
	 */
	public static double getCorrMultiple (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictor) {
		int len = columnPredictor.length;

		//count c
		RealMatrix mat = new BlockRealMatrix(dataSet.getSampleSize(), len + 1);
		for (int i = 0; i < len; i++) {
			mat.setColumn(i, dataSet.getDataMatrix().getColumn(columnPredictor[i]).arrayCopy());
		}
		mat.setColumn(len, dataSet.getDataMatrix().getColumn(columnPredicted).arrayCopy());
		RealMatrix sub = getCorrelationMatrix(mat).getColumnMatrix(len);
		RealMatrix c = new BlockRealMatrix(sub.getRowDimension() - 1, 1);
		for (int i = 0; i < sub.getRowDimension() - 1; i++) {
			c.setRow(i, sub.getRow(i));
		}
		// count Rxx
		RealMatrix mat1 = new BlockRealMatrix(dataSet.getSampleSize(), len);
		for (int i = 0; i < len; i++) {
			mat1.setColumn(i, dataSet.getDataMatrix().getColumn(columnPredictor[i]).arrayCopy());
		}

		try {
			RealMatrix rxx_1 = MatrixUtils.inverse(getCorrelationMatrix(mat1));
			// c^T R_xx^-1 c
			return sqrt(c.transpose().multiply(rxx_1).multiply(c).getEntry(0, 0));
		} catch (SingularMatrixException e) {
			return 0.0;
		}

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
	 * @param arr array
	 */
	public static boolean allEqual (double[] arr) {
		return Arrays.stream(arr).distinct().count() == 1;
	}

	/** Get deviation from median, inspired by standard deviation
	 *
	 * @param median median value
	 * @param column column
	 */
	public static double getDevMedian (double median, Vec column) {
		Vec withoutMedian = column.clone().subtract(median);
		return sqrt(withoutMedian.pairwiseMultiply(withoutMedian).sum() / column.length());
	}

	/**
	 * Calculate the min and max step difference ignoring gaps
	 *
	 */
	public static double[] calcDiffs (Vec column) {
		int n = column.length();
		double minDiff = Double.POSITIVE_INFINITY;
		double maxDiff = Double.NEGATIVE_INFINITY;
		for (int i = 1; i < n; i++) {
			double a = column.get(i);
			if (Double.isNaN(a)) { // skip next entry, since the difference cannot be counted
				i++;
				continue;
			}
			double b = column.get(i - 1);
			if (Double.isNaN(b)) {  // if there are two NaNs in a row
				continue;
			}
			double diff = abs(a - b);
			if (diff < minDiff) {
				minDiff = diff;
				continue;
			}
			if (diff > maxDiff) {
				maxDiff = diff;
			}

		}
		return new double[]{minDiff, maxDiff};
	}

	/**
	 * Checks whether the value is within the max and min of the column
	 * @param val value
	 * @param diffs array that contain min and max
	 */
	public static boolean isWithinMaxAndMin (double val, double[] diffs) {
		return Double.compare(diffs[0], val) <= 0 &&
			Double.compare(diffs[1], val) >= 0;
	}

	/** Checks whether the step from logically the closest value is within the max and min range of differences the column
	 *
	 * @param newValue predicted value
	 * @param data object that contains previous or next closest record
	 */
	public static boolean isStepWithinMaxAndMinStep (double newValue, MainData data) {
		double[] diffs = calcDiffs(data.getTrain().getNumericColumn(data.getColumnPredicted()));
		//returns true on empty stream, we cannot say it is not within step borders if there are no closest record/-s
		return Arrays.stream(data.getStep(newValue)).allMatch(i -> isWithinMaxAndMin(i, diffs));
	}

	/** Compute correlation matrix
	 *
	 * @param matrix data matrix
	 * @return correlation matrix
	 */
	public static RealMatrix getCorrelationMatrix (RealMatrix matrix) {
		int col = matrix.getColumnDimension();
		RealMatrix mat = new BlockRealMatrix(col, col);

		for (int i = 0; i < col; ++i) {
			for (int j = 0; j < i; ++j) {
				double corr = correlation(new DenseVector(matrix.getColumn(i)), new DenseVector(matrix.getColumn(j)), true);
				mat.setEntry(i, j, corr);
				mat.setEntry(j, i, corr);
			}

			mat.setEntry(i, i, 1.0D);
		}

		return mat;
	}
}