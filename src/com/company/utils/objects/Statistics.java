package com.company.utils.objects;

import jsat.SimpleDataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static com.company.utils.calculations.StatCalculations.getCorrMultiple;
import static java.lang.Math.abs;

public class Statistics {
	public int columnPredicted;
	public int[] columnPredictors;

	private double[] percentiles;
	private double mean;
	private double variance;
	private double standardDeviation;
	private double kurtosis;
	private double skewness;
	private double correlation;
	private double maxDiff;
	private double minDiff;
	private double[] thresholds;

	public Statistics (SimpleDataSet dataSet, int columnPredicted, int columnPredictor) {
		this.columnPredicted = columnPredicted;
		this.columnPredictors = new int[]{columnPredictor};
		Vec columnPredict = dataSet.getDataMatrix().getColumn(columnPredicted);
		Vec columnTrain = dataSet.getDataMatrix().getColumn(columnPredictor);

		calcBasic(removeNans(columnPredict));
		calcDiffs(removeOutliers(columnPredict, percentiles[1], percentiles[7]));

		Vec[] cols = removePairNans(columnPredict, columnTrain);
		correlation = DescriptiveStatistics.sampleCorCoeff(cols[0], cols[1]);

		calcThresholds(false);
	}

	public Statistics (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictors) {
		this.columnPredicted = columnPredicted;
		this.columnPredictors = columnPredictors;
		Vec columnPredict = dataSet.getDataMatrix().getColumn(columnPredicted);

		calcBasic(removeNans(columnPredict));
		calcDiffs(removeOutliers(columnPredict, percentiles[1], percentiles[7]));
		// TODO: might need NaNs removal
		correlation = getCorrMultiple(dataSet, columnPredicted, columnPredictors);

		calcThresholds(true);
	}

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
		//TODO: check how is it counted (question of performance)
		mean = column.mean();
		variance = column.variance();
		standardDeviation = column.standardDeviation();
		kurtosis = column.kurtosis();
		skewness = column.skewness();
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

	/**
	 * Calculate the min and max step difference ignoring gaps
	 *
	 */
	private void calcDiffs (Vec column) {
		int n = column.length();
		minDiff = Double.POSITIVE_INFINITY;
		maxDiff = Double.NEGATIVE_INFINITY;
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
	}

	/**
	 * Define thresholds for method choosing
	 * @param multiple whether there are more than one predictor
	 */
	private void calcThresholds (boolean multiple) {

		double closeMean;
		if (standardDeviation / mean > 0.75) {
			closeMean = 0.3;
		} else if (standardDeviation / mean > 0.50) {
			closeMean = 0.5;
		} else {
			closeMean = 0.7;
		}

		double closeMedian;
		if ((skewness > 1 || skewness < -1) && kurtosis < 0) {
			closeMedian = 0.5;
		} else {
			closeMedian = 0.23;
		}

		double linearRel = multiple ? 0.45 : 0.368;

		double polynomialOrder = 0.3265;

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
			"\n\tPearson Correlation Coefficient with predictor: " + correlation +
			"\n\tMin difference from previous: " + minDiff +
			"\n\tMax difference from previous: " + maxDiff +
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

	public double getCorrelation () {
		return correlation;
	}

	public double getMaxDiff () {
		return maxDiff;
	}

	public double getMinDiff () {
		return minDiff;
	}

	public double[] getThresholds () {
		return thresholds;
	}

	public double[] getPercentiles () {
		return percentiles;
	}
}