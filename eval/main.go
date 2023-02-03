package main

import (
	"bytes"
	_ "embed"
	"image"
	"image/png"
	"log"
	"os"
	"time"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

func main() {
	f, err := os.ReadFile("image.png")
	if err != nil {
		log.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(f))
	if err != nil {
		log.Fatal(err)
	}
	m, err := NewOrientationModel()
	if err != nil {
		panic(err)
	}
	t0 := time.Now()
	p, err := m.Estimate(img)
	if err != nil {
		panic(err)
	}
	t1 := time.Now()
	log.Println(t1.Sub(t0))
	log.Println(p)
}

type OrientationModel struct {
	model *onnx.Model
	backend *gorgonnx.Graph
}

//go:embed model.onnx
var onnxModel []byte

func NewOrientationModel() (*OrientationModel, error) {
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	if err := model.UnmarshalBinary(onnxModel); err != nil {
		return nil, err
	}
	return &OrientationModel{model, backend}, model.SetInput(0, tensor.New(tensor.WithShape(1, 3, 90, 160), tensor.Of(tensor.Float32)))
}

func (m *OrientationModel) Estimate(img image.Image) (float32, error) {
	// convert image to tensor
	dst := m.model.GetInputTensors()[0]
	for x := 0; x < img.Bounds().Dx(); x++ {
		for y := 0; y < img.Bounds().Dy(); y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			if err := dst.SetAt(float32(uint8(r/0x101)), 0, 0, y, x); err != nil {
				return 0, err
			}
			if err := dst.SetAt(float32(uint8(g/0x101)), 0, 1, y, x); err != nil {
				return 0, err
			}
			if err := dst.SetAt(float32(uint8(b/0x101)), 0, 2, y, x); err != nil {
				return 0, err
			}
		}
	}
	// eval model
	if err := m.backend.Run(); err != nil {
		return 0, err
	}
	// get output
	output, err := m.model.GetOutputTensors()
	if err != nil {
		return 0, err
	}
	pup, err := output[0].At(0, 0)
	if err != nil {
		return 0, err
	}
	return pup.(float32), nil
}
