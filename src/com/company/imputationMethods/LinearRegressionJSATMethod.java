package com.company.imputationMethods;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.math.SimpleLinearRegression;

import java.util.ArrayList;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class LinearRegressionJSATMethod extends ImputationMethod {
	private int columnPredictor;
	private double[] reg;

	public LinearRegressionJSATMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets, int columnPredictor) {
		super(columnPredicted, datasets);
		this.columnPredictor = columnPredictor;
	}

	public void preprocessData () {
	}

	public void fit () {
		reg = SimpleLinearRegression.regres(trainingCopy.getDataMatrix().getColumn(columnPredictor), trainingCopy.getDataMatrix().getColumn(columnPredicted));
	}

	public double predict (DataPoint dp) {
		if (Double.isNaN(reg[0]) || Double.isNaN(reg[1])) {
			return Double.NaN;
		}
		return reg[0] + reg[1] * dp.getNumericalValues().get(columnPredictor);
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "LinearRegression (columnPredictor=" + columnPredictor + ")" + ANSI_RESET + "\na & b: [" + reg[0] + "," + reg[1] + "]");
	}
}
