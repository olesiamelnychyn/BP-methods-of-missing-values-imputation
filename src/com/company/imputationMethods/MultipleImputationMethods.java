package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;

import static com.company.utils.calculations.StatCalculations.hasLinearRelationship;

public class MultipleImputationMethods {
	private SimpleDataSet datasetMissing;
	private boolean weighted;

	public MultipleImputationMethods (SimpleDataSet datasetMissing, boolean weighted) {
		this.datasetMissing = datasetMissing;
		this.weighted = weighted;
	}

	public ImputationMethod imputeMultiple (MainData data, Statistics stat) {
		DatasetManipulation.getToBeImputedAndTrainDeepCopiesByClosestDistance(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 8, weighted);
		if (data.getColumnPredictors().length == 1) {
			return null;
		}

		if (hasLinearRelationship(data.getTrain(), stat)) {
			return new MultipleLinearRegressionJSATMethod(data);
		} else {
			return new MultiDimensionalMultipleImputation(data, datasetMissing, stat);
//			return new MultipleLinearRegressionJSATMethod(columnPredicted, datasets, columnPredictors, 2);
		}
	}
}
