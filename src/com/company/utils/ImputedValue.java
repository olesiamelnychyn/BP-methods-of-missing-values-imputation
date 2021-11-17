package com.company.utils;

public class ImputedValue {
	int index;
	double actual;
	double predicted;

	public ImputedValue (int index, double actual, double predicted) {
		this.index = index;
		this.actual = actual;
		this.predicted = predicted;
	}
}
