package com.company.imputationMethods;

import com.company.utils.objects.MainData;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.Arrays;
import java.util.Collections;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class UseClosest extends ImputationMethod {
	private double closestValue;

	public UseClosest (MainData data) {
		super(data);
	}

	public void preprocessData () {
		// only the main record can be imputed this way
		data.setImpute(new SimpleDataSet(Collections.singletonList(data.getDp())));
		toBePredicted = data.getImpute();
	}

	public void fit () {
		closestValue = Arrays.stream(data.getToCountStepWith())
			.filter(Objects::nonNull)
			.collect(Collectors.toList())
			.get(0).getNumericalValues().get(data.getColumnPredicted());
	}

	public double predict (DataPoint dp) {
		return closestValue;
	}

	@Override
	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "Closest Imputation" + ANSI_RESET + "\nClosest: [" + closestValue + "]");
	}
}
