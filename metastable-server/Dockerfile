ARG TARGET=x86_64-unknown-linux-musl

FROM rust:1-slim-bullseye AS build

ARG TARGET

# Install rust target and musl dependencies
RUN rustup target add ${TARGET}
RUN apt update && \
    apt install -y \
      musl-dev \
      musl-tools \
      linux-libc-dev

WORKDIR /usr/src

# Fetch dependencies in a single layer for better cacheability.
COPY Cargo.toml Cargo.lock ./

RUN mkdir src && \
    echo 'fn main() {}' > src/main.rs && \
    cargo fetch

# Copy the source and build the executable.
COPY . .

RUN cargo build \
      --target ${TARGET} \
      --release

FROM gcr.io/distroless/base-debian11

ARG TARGET
 
COPY --from=build /usr/src/target/${TARGET}/release/metastable-server /metastable-server
 
ENTRYPOINT ["/metastable-server"]
