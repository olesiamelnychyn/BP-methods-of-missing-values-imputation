package com.company.utils.calculations;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import static com.company.utils.calculations.MathCalculations.*;
import static java.lang.Math.abs;
import static jsat.linear.distancemetrics.PearsonDistance.correlation;

public class StatCalculations {

	static public boolean isStrictlyIncreasing (SimpleDataSet dataSet, int column) {
		double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
		for (int i = 1; i < dataSet.getSampleSize(); i++) {
			if (dataSet.getDataPoint(i).getNumericalValues().get(column) < current) {
				return false;
			}
			current = dataSet.getDataPoint(i).getNumericalValues().get(column);
		}
		return true;
	}

	static public boolean isStrictlyDecreasing (SimpleDataSet dataSet, int column) {
		double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
		for (int i = 1; i < dataSet.getSampleSize(); i++) {

			if (dataSet.getDataPoint(i).getNumericalValues().get(column) > current) {
				return false;
			}
			current = dataSet.getDataPoint(i).getNumericalValues().get(column);
		}
		return true;
	}

	static public boolean isCloseToMean (SimpleDataSet dataSet, int columnPredicted, double threshold) {
		double std = dataSet.getDataMatrix().getColumn(columnPredicted).standardDeviation();
		double mean = dataSet.getDataMatrix().getColumn(columnPredicted).mean();
		if (std / mean <= threshold) {
			return true;
		}
		return false;
	}

	static public boolean isCloseToMedian (SimpleDataSet dataSet, int columnPredicted, double threshold) {
		double dist = 0.0;
		double median = dataSet.getDataMatrix().getColumn(columnPredicted).median();
		for (DataPoint dp : dataSet.getDataPoints()) {
			dist += abs(dp.getNumericalValues().get(columnPredicted) - median);
		}
		if (dist / 8 / median <= threshold) {
			return true;
		}
		return false;
	}

	//calculate pearson correlation with one predictor
	static public boolean hasLinearRelationship (SimpleDataSet dataSet, int columnPredicted, int columnPredictor, double threshold) {
		double corr = correlation(dataSet.getNumericColumn(columnPredicted), dataSet.getNumericColumn(columnPredictor), true);
		if (abs(corr) < threshold) {
			return false;
		}
		return true;
	}

	//calculate pearson correlation with multiple predictors
	static public boolean hasLinearRelationship (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictor, double threshold) {
		if (getCorrMultiple(dataSet, columnPredicted, columnPredictor) > threshold) {
			return false;
		}
		return true;
	}

	static public int getPolynomialOrder (SimpleDataSet dataSet, int columnPredicted, int columnPredictor, double threshold) {
		int n = dataSet.getSampleSize();
		//create raw polynomials of values in columnPredictor
		double[][] powPred = new double[4][n];
		for (int pow = 0; pow < 4; pow++) {
			for (int index = 0; index < n; index++) {
				double x = dataSet.getDataPoint(index).getNumericalValues().get(columnPredictor);
				powPred[pow][index] = getPolyWithoutCoefficients(x, pow);
			}
		}
		// calculate correlations between raw polynomials and values in columnPredicted
		double[] corr = new double[4];
		for (int pow = 1; pow < 4; pow++) {
			corr[pow] = correlation(dataSet.getNumericColumn(columnPredicted), new DenseVector(powPred[pow]), true);
		}
		int iMax = getMax(corr); //choose the best correlation
		if (abs(corr[iMax]) < threshold) {
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
	 * @return
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

}