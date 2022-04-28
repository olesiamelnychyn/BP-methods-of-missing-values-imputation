package com.company.utils.calculations;

import jsat.classifiers.DataPoint;
import jsat.linear.Vec;

import java.util.Arrays;
import java.util.List;

import static java.lang.Math.*;

/**
 * Class for common math calculations
 */
public class MathCalculations {

	public static double polyValue (double[] p, double x0) {

		if (p == null) {
			return Double.NaN;
		}
		if (p.length < 1) {
			return 0.0;
		}

		double val = p[p.length - 1];

		for (int i = p.length - 2; i >= 0; i--) {
			val = val * x0 + p[i];
		}
		return val;
	}

	public static double gaussianValue (double[] p, double x0) {

		if (p == null) {
			return Double.NaN;
		}
		if (p.length < 1) {
			return 0.0;
		}

		double x = (x0 - p[1]) / p[2];
		return p[0] * Math.exp(x * x / -2);
	}

	static public double getPolyWithoutCoefficients (double x, int power) {
		double y = 0.0;
		for (int i = 0; i < power; i++) {
			y += pow(x, i);
		}
		return y;
	}

	static public int getMax (double[] x) {
		int index = 0;
		double max = x[0];
		for (int i = 1; i < x.length; i++) {
			if (abs(x[i]) > abs(max)) {
				max = x[i];
				index = i;
			}
		}
		return index;
	}

	static public int[] getIndexesOfNull (DataPoint dp) {
		Vec vec = dp.getNumericalValues();
		if (vec.countNaNs() == 0) {
			return new int[0];
		}
		int[] indexes = new int[vec.countNaNs()];
		int j = 0;
		for (int i = 0; i < vec.length(); i++) {
			if (Double.isNaN(vec.get(i))) {
				indexes[j++] = i;
			}
		}
		return indexes;
	}

	public static int[] getIntersection (int[] arr1, int[] arr2) {
		return Arrays.stream(arr1)
			.distinct()
			.filter(x -> Arrays.stream(arr2).anyMatch(y -> y == x))
			.toArray();
	}

	public static int[] getDifference (int[] arr1, int[] arr2) {
		return Arrays.stream(arr1)
			.distinct()
			.filter(x -> Arrays.stream(arr2).noneMatch(y -> y == x))
			.toArray();
	}

	public static double getEuclideanDistance (double[] dp1, double[] dp2) {
		double sum = 0.0;
		int n = dp1.length;
		for (int i = 0; i < n; i++) {
			double diff = dp1[i] - dp2[i];
			sum += diff * diff;
		}

		return sqrt(sum);
	}

	public static double getWeightByEuclidean (double dist) {
		return 1 / (dist + 0.01);
	}

	/**
	 * Normalize weight of data points, so they sum up to 1
	 */
	public static void normalizeWeights (List<DataPoint> dataPoints) {
		double sum = dataPoints
			.stream()
			.map(DataPoint::getWeight)
			.reduce(0.0, Double::sum);
		if (sum > 0.0) {
			dataPoints.forEach(dp -> dp.setWeight(dp.getWeight() / sum));
		}
	}

	/**
	 * Normalize one value
	 * @param x value
	 * @param max max in set
	 * @param min min in set
	 * @param n number of values in set
	 */
	public static double normalize (double x, double max, double min, double n) {
		if (Double.compare(min, max) == 0) {
			return 1 / n;
		}
		return (x - min) / (max - min);
	}

	/** Normalize predictors columns to be within the range from 0 to 1
	 *
	 * @param dp
	 * @param columnPredictors
	 * @param sumsForNormalisation
	 * @param size
	 * @return
	 */
	public static double[] normalizePredictors (DataPoint dp, int[] columnPredictors, double[] sumsForNormalisation, int size) {
		double[] values = new double[columnPredictors.length];
		for (int i = 0; i < columnPredictors.length; i++) {
			values[i] = normalize(dp.getNumericalValues().get(columnPredictors[i]), sumsForNormalisation[i * 2], sumsForNormalisation[i * 2 + 1], size);
		}
		return values;
	}
}