package com.company.utils.objects;

import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static com.company.utils.ColorFormatPrint.*;
import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

public class PerformanceMeasures {
	private Vec actual;
	private Vec predicted;
	private double meanTraining;
	protected double[] measures;
	private int columnIndex;

	public PerformanceMeasures (Vec actual, Vec predicted, double meanTraining, int columnIndex) {
		this.actual = actual;
		this.predicted = predicted;
		this.meanTraining = meanTraining;
		this.columnIndex = columnIndex;
		calcMeasures();
	}

	public double[] getMeasures () {
		return measures;
	}

	/**
	 * Calculates performance measures and saves into a predefined array
	 */
	private void calcMeasures () {
		measures = new double[]{
			MSError(actual, predicted),
			RMSError(actual, predicted),
			meanAbsoluteError(actual, predicted),
			relativeSquaredError(actual, predicted, meanTraining),
			rootRelativeSquaredError(actual, predicted, meanTraining),
			relativeAbsoluteError(actual, predicted, meanTraining),
			DescriptiveStatistics.sampleCorCoeff(actual, predicted),
			meanAbsolutePercentageError(actual, predicted)
		};
	}

	/** Prints out to the console and writes to output file performance measures
	 *
	 * @throws IOException
	 */
	public void printAndWriteResults () throws IOException {
		String str = toString();
		printPerformanceMeasures(str);
		writeOutputPerformanceMeasures(str);
	}

	public String toString () {
		return "\n\n\tMean-Squared Error: " + format(measures[0]) +
			"\n\tRoot Mean-Squared Error: " + format(measures[1]) +
			"\n\tMean-Absolute Error: " + format(measures[2]) +
			"\n\tRelative-Squared Error: " + format(measures[3]) + "%" +
			"\n\tRoot Relative-Squared Error: " + format(measures[4]) + "%" +
			"\n\tRelative-Absolute Error: " + format(measures[5]) + "%" +
			"\n\tPearson Correlation Coefficient: " + format(measures[6]) +
			"\n\tMean Absolute Percentage Error: " + format(measures[7]) + "%\n\n";
	}

	/**
	 * Print out performance measures
	 * @param str names of the metrics + values
	 */
	public void printPerformanceMeasures (String str) {
		System.out.print("Performance (Predictions for column " + ANSI_BOLD_ON + ANSI_PURPLE + columnIndex + ANSI_RESET + ANSI_BOLD_OFF + "):");
		System.out.print(ANSI_BOLD_ON + ANSI_PURPLE + str + ANSI_RESET + ANSI_BOLD_OFF);
	}

	/**
	 * Write performance measures to the output file for statistics
	 * @param str names of the metrics + values
	 */
	public void writeOutputPerformanceMeasures (String str) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt", true));
		String output = "\nPerformance (Predictions for column: " + columnIndex + "):";
		writer.append(output).append(str);
		writer.close();
	}

	/** Mean-Squared Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @return mean-squared error
	 */
	static public double MSError (Vec actual, Vec predicted) {
		int n = actual.length();
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			double x = predicted.get(i) - actual.get(i);
			sum += x * x;
		}

		return sum / n;
	}

	/** Root Mean-Squared Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @return root mean-squared error
	 */
	static public double RMSError (Vec actual, Vec predicted) {
		return sqrt(MSError(actual, predicted));
	}

	/** Mean-Absolute Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @return mean-absolute error
	 */
	static public double meanAbsoluteError (Vec actual, Vec predicted) {
		int n = actual.length();
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			sum += abs(predicted.get(i) - actual.get(i));
		}

		return sum / n;
	}

	/** Relative-Squared Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @param meanTraining mean over training dataset
	 * @return relative-squared error
	 */
	static public double relativeSquaredError (Vec actual, Vec predicted, double meanTraining) {
		int n = actual.length();
		double sumTop = 0.0, sumBottom = 0.0;
		for (int i = 0; i < n; i++) {
			double x = predicted.get(i) - actual.get(i);
			sumTop += x * x;
			x = actual.get(i) - meanTraining;
			sumBottom += x * x;
		}
		return (sumTop / sumBottom) * 100;
	}

	/** Root Relative-Squared Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @param meanTraining mean over training dataset
	 * @return root relative-squared error
	 */
	static public double rootRelativeSquaredError (Vec actual, Vec predicted, double meanTraining) {
		int n = actual.length();
		double sumTop = 0, sumBottom = 0;
		for (int i = 0; i < n; i++) {
			double x = predicted.get(i) - actual.get(i);
			sumTop += x * x;
			x = (actual.get(i) - meanTraining);
			sumBottom += x * x;
		}

		return sqrt(sumTop / sumBottom) * 100;
	}

	/** Relative-Absolute Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @param meanTraining mean over training dataset
	 * @return relative-absolute error
	 */
	static public double relativeAbsoluteError (Vec actual, Vec predicted, double meanTraining) {
		int n = actual.length();
		double sumTop = 0.0, sumBottom = 0.0;
		for (int i = 0; i < n; i++) {
			sumTop += abs(predicted.get(i) - actual.get(i));
			sumBottom += abs(actual.get(i) - meanTraining);
		}


		return sumTop / sumBottom * 100;
	}

	/** Mean Absolute Percentage Error
	 *
	 * @param predicted predicted values
	 * @param actual original values
	 * @return relative-absolute error
	 */
	static public double meanAbsolutePercentageError (Vec actual, Vec predicted) {
		int n = actual.length();
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			// handle zero case
			if (Double.compare(actual.get(i), 0.0) == 0) {
				sum += abs(predicted.get(i) / (0.0 + actual.mean() * 0.01));
			} else {
				sum += abs((actual.get(i) - predicted.get(i)) / actual.get(i));
			}
		}
		return sum * 100 / n;
	}
}
