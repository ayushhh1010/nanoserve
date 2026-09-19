package router

import (
	"fmt"
	"hash/fnv"
	"sort"
	"testing"
)

// TestPrintHashDistribution is a measurement, not an assertion. It reproduces
// the load distribution with and without the fmix64 finalizer so the numbers
// quoted elsewhere come from a run rather than from memory.
//
//	go test . -run TestPrintHashDistribution -v
func TestPrintHashDistribution(t *testing.T) {
	const (
		replicas = 8
		vnodes   = 150
		keys     = 400
	)

	// rawHashKey is hashKey as it was before the fix: FNV-1a with no
	// avalanche finalizer.
	rawHashKey := func(s string) uint64 {
		h := fnv.New64a()
		_, _ = h.Write([]byte(s))
		return h.Sum64()
	}

	distribute := func(hash func(string) uint64) []int {
		type pt struct {
			h   uint64
			rep int
		}
		var ring []pt
		for r := 0; r < replicas; r++ {
			for v := 0; v < vnodes; v++ {
				ring = append(ring, pt{hash(fmt.Sprintf("replica-%d#%d", r, v)), r})
			}
		}
		sort.Slice(ring, func(i, j int) bool { return ring[i].h < ring[j].h })

		counts := make([]int, replicas)
		for k := 0; k < keys; k++ {
			// Short, similar keys: exactly the traffic shape this serves.
			h := hash(fmt.Sprintf("key-%d", k))
			i := sort.Search(len(ring), func(i int) bool { return ring[i].h >= h })
			if i == len(ring) {
				i = 0
			}
			counts[ring[i].rep]++
		}
		return counts
	}

	report := func(label string, c []int) {
		max, min, zeros := 0, c[0], 0
		for _, v := range c {
			if v > max {
				max = v
			}
			if v < min {
				min = v
			}
			if v == 0 {
				zeros++
			}
		}
		t.Logf("%s: %v  max=%d (%.1f%%) min=%d empty=%d",
			label, c, max, 100*float64(max)/float64(keys), min, zeros)
	}

	report("FNV-1a alone  ", distribute(rawHashKey))
	report("with fmix64   ", distribute(hashKey))
}
