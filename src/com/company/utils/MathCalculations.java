package com.company.utils;

import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import java.util.Arrays;
import static java.lang.Math.pow;

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
			if (x[i] > max) {
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
}
