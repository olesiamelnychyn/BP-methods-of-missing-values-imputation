package com.company.imputationMethods;

import com.company.utils.objects.MainData;
import jsat.classifiers.DataPoint;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MeanImputationMethod extends ImputationMethod {
	private double mean;

	public MeanImputationMethod (MainData data) {
		super(data);
	}

	public void preprocessData () {
	}

	public void fit () {
		mean = trainingCopy.getDataMatrix().getColumn(columnPredicted).mean();
	}

	public double predict (DataPoint dp) {
		return mean;
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "Mean Imputation" + ANSI_RESET + "\nMean: [" + mean + "]");
	}
}
