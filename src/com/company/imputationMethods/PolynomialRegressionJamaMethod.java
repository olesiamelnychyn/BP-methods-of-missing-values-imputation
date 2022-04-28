package com.company.imputationMethods;

import com.company.imputationMethods.regressions.PolynomialRegression;
import com.company.utils.objects.MainData;
import jsat.classifiers.DataPoint;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class PolynomialRegressionJamaMethod extends ImputationMethod {
	private int columnPredictor;
	private PolynomialRegression polynomialRegression;

	public PolynomialRegressionJamaMethod (MainData data) {
		super(data);
		this.columnPredictor = data.getColumnPredictors()[0];
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
		System.out.println(IntStream.range(0, polynomialRegression.degree())
			.mapToDouble(polynomialRegression::beta)
			.mapToObj(String::valueOf)
			.collect(Collectors.joining(", ", "Coefficients: [", "]")));
	}
}
