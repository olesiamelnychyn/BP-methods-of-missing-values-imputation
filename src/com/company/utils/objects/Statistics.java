package com.company.utils.objects;

import jsat.SimpleDataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;

import java.util.Arrays;

import static com.company.utils.calculations.StatCalculations.getCorrMultiple;
import static java.lang.Math.abs;

public class Statistics {
	private double min;
	private double max;
	private double mean;
	private double median;
	private double variance;
	private double standardDeviation;
	private double kurtosis;
	private double skewness;
	private double correlation;
	private double maxDiff;
	private double minDiff;
	private double[] thresholds;

	public Statistics (Vec columnPredicted, Vec columnPredictor) {
		Vec column = removeNans(columnPredicted);
		Vec[] cols = removePairNans(columnPredicted, columnPredictor);

		calcBasic(column);
		calcDiffs(columnPredicted);
		correlation = DescriptiveStatistics.sampleCorCoeff(cols[0], cols[1]);

		calcThresholds(false);
	}

	public Statistics (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictor) {
		Vec columnPredict = dataSet.getDataMatrix().getColumn(columnPredicted);
		Vec column = removeNans(columnPredict);

		calcBasic(column);
		calcDiffs(columnPredict);
		// TODO: might need NaNs removal
		correlation = getCorrMultiple(dataSet, columnPredicted, columnPredictor);

		calcThresholds(true);
	}

	private void calcBasic (Vec column) {
		min = column.min();
		max = column.max();
		mean = column.mean();
		median = column.median();
		variance = column.variance();
		standardDeviation = column.standardDeviation();
		kurtosis = column.kurtosis();
		skewness = column.skewness();
	}

	/**
	 * Remove entries which contain NaN
	 *
	 * @param column
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
	 * Remove entries at the same index from both vectors if at least one contains NaN
	 *
	 * @param column1
	 * @param column2
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
	 * @param column
	 */
	private void calcDiffs (Vec column) {
		int n = column.length();
		minDiff = abs(max);
		maxDiff = abs(min);
		for (int i = 1; i < n; i++) {
			double a = column.get(i);
			if (Double.isNaN(a)) { // skip next entry, since the difference cannot be count
				i++;
				continue;
			}
			double b = column.get(i - 1);
			if (Double.isNaN(b)) {  // if there are two NaNs in a row
				continue;
			}
			double diff = abs(a - b);
			if (diff < minDiff) {
				if (diff == 0) {
					System.out.println("Diff: " + a + " " + b);
				}
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
		thresholds = new double[]{
				0.41, // for isCloseToMean()
				0.23, // for isCloseToMedian()
				multiple ? 0.45 : 0.368, // for hasLinearRelationship()
				0.3265, // for getPolynomialOrder()
		};
	}


	public String toString () {
		return "Statistics of predicted value:" +
				"\n\tMin: " + min +
				"\n\tMax: " + max +
				"\n\tMean: " + mean +
				"\n\tMedian: " + median +
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

	public double getMin () {
		return min;
	}

	public double getMax () {
		return max;
	}

	public double getMean () {
		return mean;
	}

	public double getMedian () {
		return median;
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

}