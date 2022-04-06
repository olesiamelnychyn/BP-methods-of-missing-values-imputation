package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;

import static com.company.utils.calculations.StatCalculations.*;

public class MultipleImputationMethods {
	private SimpleDataSet datasetMissing;
	private boolean weighted;
	private boolean useMultiDimensionalAsDefault;

	public MultipleImputationMethods (SimpleDataSet datasetMissing, boolean weighted, boolean useMultiDimensionalAsDefault) {
		this.datasetMissing = datasetMissing;
		this.weighted = weighted;
		this.useMultiDimensionalAsDefault = useMultiDimensionalAsDefault;
	}

	public ImputationMethod imputeMultiple (MainData data, Statistics stat) {
		// prepare datasets
		DatasetManipulation.getToBeImputedAndTrainDeepCopiesByClosestDistance(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 12, weighted);
		if (data.getColumnPredictors().length == 1) {
			// After composing train dataset some predictor columns contain same values, such columns' indexes are excluded from predictors.
			// In case only one predictor left return null, so the program can continue with single imputation method.
			return null;
		}

		// select appropriate imputation method
		if (hasLinearRelationship(data, stat)) {
			// linear regression is a special way of polynomial regression with 1st degree
			return new MultiplePolynomialRegressionJSATMethod(data);
		} else if (hasPolynomialRelation(data, stat)) {
			return new MultiplePolynomialRegressionJSATMethod(data, 2);
		} else if (useMultiDimensionalAsDefault) {
			return new MultiDimensionalMultipleImputation(data, datasetMissing, stat);
		}
		return new MeanImputationMethod(data);
	}
}
