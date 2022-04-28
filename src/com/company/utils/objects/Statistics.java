package com.company.utils.objects;

import com.company.utils.calculations.StatCalculations;
import jsat.SimpleDataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static com.company.utils.DatasetManipulation.removeNanRowsByColumns;
import static com.company.utils.calculations.StatCalculations.getCorrMultiple;
import static com.company.utils.calculations.StatCalculations.getDevMedian;

/**
 * Class that holds a statistics of the predicted column
 */
public class Statistics {
	public int columnPredicted;
	public int[] columnPredictors;
	private double[] percentiles;
	private double mean;
	private double variance;
	private double standardDeviation;
	private double kurtosis;
	private double skewness;
	private Map<Integer, Double> correlationSimple = new HashMap<>();
	private double correlationMultiple = Double.NaN;
	private double[] diffs; // minDiff and maxDiff
	private double[] thresholds;
	private double devMedian;

	// statistics for simple imputation
	public Statistics (SimpleDataSet dataSet, int columnPredicted, int columnPredictor) {
		this.columnPredicted = columnPredicted;
		this.columnPredictors = new int[]{columnPredictor};
		calculateCorrelationSimple(dataSet, columnPredictor);
		calcThresholds(false);
	}

	// statistics for multiple imputation
	public Statistics (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictors) {
		this.columnPredicted = columnPredicted;
		this.columnPredictors = columnPredictors;
		Vec columnPredict = dataSet.getDataMatrix().getColumn(columnPredicted);

		calcBasic(removeNans(columnPredict));
		diffs = StatCalculations.calcDiffs(removeOutliers(columnPredict, percentiles[0], percentiles[8]));

		calcCorr(dataSet);
		calcThresholds(true);
	}

	// basic statistics, which is common for both simple and multiple imputation cases
	private void calcBasic (Vec column) {
		List<Double> latencies = DoubleStream.of(column.arrayCopy()).boxed().sorted().collect(Collectors.toCollection(ArrayList::new));

		percentiles = new double[]{
			getPercentile(latencies, 0),
			getPercentile(latencies, 15),
			getPercentile(latencies, 25),
			getPercentile(latencies, 35),
			getPercentile(latencies, 50),
			getPercentile(latencies, 65),
			getPercentile(latencies, 75),
			getPercentile(latencies, 85),
			getPercentile(latencies, 100)
		};
		mean = column.mean();
		variance = column.variance();
		standardDeviation = column.standardDeviation();
		kurtosis = column.kurtosis();
		skewness = column.skewness();
		devMedian = getDevMedian(percentiles[4], column);
	}

	/** Calculate correlation between predicted column and the column passed
	 *
	 * @param dataSet data
	 * @param columnIndex column to calculate the correlation with
	 */
	private void calculateCorrelationSimple (SimpleDataSet dataSet, int columnIndex) {
		Vec columnPredict = dataSet.getDataMatrix().getColumn(columnPredicted);
		Vec columnTrain = dataSet.getDataMatrix().getColumn(columnIndex);

		calcBasic(removeNans(columnPredict));

		diffs = StatCalculations.calcDiffs(removeOutliers(columnPredict, percentiles[0], percentiles[8]));

		Vec[] cols = removePairNans(columnPredict, columnTrain);
		correlationSimple.put(columnIndex, DescriptiveStatistics.sampleCorCoeff(cols[0], cols[1]));
	}

	/**
	 * Remove entries which contain NaN
	 *
	 * @return vector clean from NaNs
	 */
	private DenseVector removeNans (Vec column) {
		return new DenseVector(
			Arrays.stream(column.arrayCopy())
				.filter(x -> !Double.isNaN(x))
				.toArray()
		);
	}

	/**
	 * Remove entries below percentile1 and above percentile2
	 *
	 * @return vector with values from column between percentile1 and percentile2
	 */
	private DenseVector removeOutliers (Vec column, double percentile1, double percentile2) {
		return new DenseVector(
			Arrays.stream(column.arrayCopy())
				.filter(x -> x > percentile1 && x < percentile2)
				.toArray()
		);
	}

	/**
	 * Remove entries at the same index from both vectors if at least one contains NaN
	 *
	 * @return two clean from NaNs vectors
	 */
	private DenseVector[] removePairNans (Vec column1, Vec column2) {
		int n = 0;
		int k = column1.length();
		double[] col1 = new double[k];
		double[] col2 = new double[k];

		for (int i = 0; i < k; i++) {
			if (!Double.isNaN(column1.get(i)) && !Double.isNaN(column2.get(i))) {
				col1[n] = column1.get(i);
				col2[n++] = column2.get(i);
			}
		}

		return new DenseVector[]{
			new DenseVector(Arrays.copyOfRange(col1, 0, n)),
			new DenseVector(Arrays.copyOfRange(col2, 0, n))
		};
	}

	/** Calculate:
	 * - multiple correlation coefficient between predicted column and predictors
	 * - Pearson correlation coefficient between predicted column and each predictor separately
	 *
	 * @param dataSet
	 */
	private void calcCorr (SimpleDataSet dataSet) {
		int[] cols = IntStream.concat(IntStream.of(columnPredictors), IntStream.of(columnPredicted)).toArray();
		correlationMultiple = getCorrMultiple(removeNanRowsByColumns(dataSet, cols), columnPredicted, columnPredictors);
		for (int predictor : columnPredictors) {
			calculateCorrelationSimple(dataSet, predictor);
		}
	}

	/**
	 * Define thresholds for method choosing
	 * @param multiple whether there are more than one predictor
	 */
	private void calcThresholds (boolean multiple) {

		double closeMean;
		if (standardDeviation / mean > 0.75) {
			closeMean = multiple ? 0.1 : 0.3;
		} else if (standardDeviation / mean > 0.50) {
			closeMean = multiple ? 0.1 : 0.5;
		} else {
			closeMean = multiple ? 0.05 : 0.1;
		}

		double closeMedian;
		if (devMedian / percentiles[4] > 0.75) {
			closeMedian = 0.4;
		} else if (devMedian / percentiles[4] > 0.50) {
			closeMedian = multiple ? 0.2 : 0.07;
		} else {
			closeMedian = multiple ? 0.05 : 0.07;
		}

		double linearRel = multiple ? 0.9 : (skewness < -0.4 ? 0.8 : 0.6);
		double polynomialOrder = multiple ? 0.81 : (skewness < -0.4 ? 0.8 : 0.7);

		thresholds = new double[]{
			closeMean, // for isCloseToMean()
			closeMedian, // for isCloseToMedian()
			linearRel, // for hasLinearRelationship()
			polynomialOrder, // for getPolynomialOrder()
		};
	}

	public static double getPercentile (List<Double> latencies, double percentile) {
		int index = (int) Math.ceil(percentile / 100.0 * latencies.size());
		return latencies.get(index == 0 ? index : index - 1);
	}

	public String toString () {
		StringBuilder correlations = new StringBuilder();
		if (!Double.isNaN(correlationMultiple)) {
			correlations.append("\n\tCorrelation Multiple: " + correlationMultiple);
		}
		for (int pred : columnPredictors) {
			correlations.append("\n\tPearson Correlation Coefficient with predictor (column " + pred + "): " + correlationSimple.getOrDefault(pred, 0.0));
		}

		return "Statistics of predicted value:" +
			"\n\tMean: " + mean +
			"\n\tMin (0th percentile): " + percentiles[0] +
			"\n\t15th percentile: " + percentiles[1] +
			"\n\t25th percentile: " + percentiles[2] +
			"\n\t35th percentile: " + percentiles[3] +
			"\n\tMedian (50th percentile): " + percentiles[4] +
			"\n\t65th percentile: " + percentiles[5] +
			"\n\t75th percentile: " + percentiles[6] +
			"\n\t85th percentile: " + percentiles[7] +
			"\n\tMax (100th percentile): " + percentiles[8] +
			"\n\tVariance: " + variance +
			"\n\tStandard deviation: " + standardDeviation +
			"\n\tKurtosis: " + kurtosis +
			"\n\tSkewness: " + skewness +
			correlations +
			"\n\tMin difference from previous: " + diffs[0] +
			"\n\tMax difference from previous: " + diffs[1] +
			"\nThresholds:" +
			"\n\tFor isCloseToMean(): " + thresholds[0] +
			"\n\tFor isCloseToMedian(): " + thresholds[1] +
			"\n\tFor hasLinearRelationship(): " + thresholds[2] +
			"\n\tFor getPolynomialOrder(): " + thresholds[3] +
			"\n";
	}

	public double getMean () {
		return mean;
	}

	public double getVariance () {
		return variance;
	}

	public double getStandardDeviation () {
		return standardDeviation;
	}

	public double getKurtosis () {
		return kurtosis;
	}

	public double getSkewness () {
		return skewness;
	}

	public Map<Integer, Double> getCorrelationSimple () {
		return correlationSimple;
	}

	public double getCorrelationSimpleByColumn (int column) {
		return correlationSimple.getOrDefault(column, 0.0);
	}

	public double getCorrelationMultiple () {
		return correlationMultiple;
	}

	public double[] getDiffs () {
		return diffs;
	}

	public double[] getMinAndMax () {
		return new double[]{percentiles[0], percentiles[8]};
	}

	public double[] getThresholds () {
		return thresholds;
	}

	public double[] getPercentiles () {
		return percentiles;
	}
}