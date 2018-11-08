package main

import (
	"bufio"
	"encoding/csv"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"
)

// A species is identified (labeled) by a number
type species int

// flowers is a slice of flowers
type flowers []flower

const (
	unknown    species = iota // 0
	setosa                    // 1
	versicolor                // 2
	virginica                 // 3
)

// A flower is a set of values and a label (species).
type flower struct {
	// label indicates what species the flower belongs to
	label species
	// values are numeric data describing the flower's featurs
	values []float64
}

// getFlowers gets flower data from a csv file.
func getFlowers(filename string) (flowers, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	var flowers flowers
	var line []string
	var spec species
	vals := make([]float64, 4)
	counter := 0
	for {
		line, err = reader.Read()
		counter++
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		for i := range vals {
			vals[i], err = strconv.ParseFloat(line[i], 64)
			if err != nil {
				return nil, err
			}
		}
		switch line[4] {
		case "Iris-setosa":
			spec = setosa
		case "versicolor":
			spec = versicolor
		case "virginica":
			spec = virginica
		default:
			spec = unknown
		}
		flowers = append(flowers, flower{label: spec, values: vals})
	}
	return flowers, nil
}

// shuffle returns a shuffled deep copy of flowers.
func shuffle(f flowers) flowers {
	sf := make(flowers, len(f))
	copy(sf, f)
	rand.Seed(int64(time.Now().Second()))
	rand.Shuffle(sf.Len(), sf.Swap)
	return sf
}

// sortFlowers sorts (stable) a deep copy of flowers by species.
func sortFlowers(f flowers) flowers {
	sf := make(flowers, len(f))
	copy(sf, f)
	sort.Stable(sf)
	return sf
}

// Len returns the number of flowers.
func (f flowers) Len() int {
	return len(f)
}

// Less returns the less-than comparison of two flowers.
func (f flowers) Less(i, j int) bool {
	return f[i].label < f[j].label
}

// Swap swaps two flowers.
func (f flowers) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

// trainer: TODO
func trainer(fs flowers) float64 {
	return 0
}
