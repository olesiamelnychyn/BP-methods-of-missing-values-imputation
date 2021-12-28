package com.company.imputationMethods;

import com.company.imputationMethods.regressions.PolynomialRegression;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class PolynomialRegressionJamaMethod extends ImputationMethod {
	private int columnPredictor;
	private PolynomialRegression polynomialRegression;

	public PolynomialRegressionJamaMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets, int columnPredictor) {
		super(columnPredicted, datasets);
		this.columnPredictor = columnPredictor;
	}

	public void preprocessData () {
	}

	public void fit () {
		polynomialRegression = new PolynomialRegression(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy(), 2);
	}

	public double predict (DataPoint dp) {
		return polynomialRegression.predict(dp.getNumericalValues().get(columnPredictor));
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "PolynomialRegressionJama (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
		System.out.print("Coefficients: [");
		for (int i = 0; i <= polynomialRegression.degree() - 1; i++) {
			System.out.print(polynomialRegression.beta(i) + ",");
		}
		System.out.println(polynomialRegression.beta(polynomialRegression.degree()) + "]");
	}
}