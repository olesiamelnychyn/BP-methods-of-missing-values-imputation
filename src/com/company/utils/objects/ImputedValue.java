package com.company.utils.objects;

/**
 * Class which contains data used for evaluation about one value
 */
public class ImputedValue {
	public int index;
	public double actual;
	public double predicted;

	public ImputedValue (int index, double actual, double predicted) {
		this.index = index;
		this.actual = actual;
		this.predicted = predicted;
	}

	public String toString () {
		return index + ": " + actual + " --- " + predicted;
	}
}
