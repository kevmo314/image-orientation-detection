FROM --platform=arm64 golang:1.19.4-bullseye AS build

WORKDIR /go/src/eval

COPY . .

RUN go mod download

RUN go build -o /main main.go

CMD [ "/main" ]

