# How to Build from Source

This guide provides step-by-step instructions for building the Video Accuracy Evaluation service from source.

If you want to build the microservices image locally, you can optionally refer to the steps in the [Building the Backend Image](#building-the-backend-image) and [Building the UI Image](#building-the-ui-image) sections. These sections provide detailed instructions on how to build the Docker images for both the backend and UI components of the `Video Accuracy Evaluation` service separately.

If you want to build the images via `docker compose`, please refer to the section [Build the Images via Docker Compose](#build-the-images-via-docker-compose).

Once all the images are built, you can proceed to start the service using the `docker compose` command as described in the [Get Started](./get-started.md) page.

## Building the Backend Image
To build the Docker image for the evaluation service, follow these steps:

1. Ensure you are in the project directory:

   ```bash
   cd microservices/video-accuracy-evaluation
   ``` 

2. Build the docker image using command:

   ```bash
   docker build -t vss-eval:latest -f docker/Dockerfile .
   ```

3. Verify that the Docker image has been built successfully:

   ```bash
   docker images | grep vss-eval
   ```

   You should see an entry for `vss-eval` with the `latest` tag.

## Building the UI image
To build the Docker image for the `vss-eval-ui` service, follow these steps:

1. Ensure you in the `ui/` project directory:

   ```bash
   cd microservices/video-accuracy-evaluation/ui
   ```

2. Build the Docker image using the provided `Dockerfile`:

   ```bash
   docker build -t vss-eval-ui:latest -f docker/Dockerfile .
   ```

3. Verify that the Docker image has been built successfully:

   ```bash
   docker images | grep vss-eval-ui
   ```

4. Once you have verified that the image has been built successfully, navigate back to the `video-accuracy-evaluation` directory:

   ```bash
   cd ..
   ```

## Build the Images via Docker Compose
This guide explains how to build the images using the compose.yaml file via the `docker compose` command.

1. Ensure you are in the project directory:

   ```bash
   cd microservices/video-accuracy-evaluation
   ```

2. Set Up Environment Variables:

   ```bash
   export HUGGINGFACEHUB_API_TOKEN=<your-huggingface-token>
   source scripts/setup_env.sh
   ```

3. Build the Docker images defined in the `compose.yaml` file:

   ```bash
   docker compose -f docker/compose.yaml build
   ```

4. Verify that the Docker images have been built successfully:

   ```bash
   docker images | grep vss-eval
   ```

   You should see entries for both `vss-eval` and `vss-eval-ui`.

## Running the Application Container
After building the images for the `Video Accuracy Evaluation` application, you can run the application container using `docker compose` by following these steps:

1. Set Up Environment Variables:

   ```bash
   export HUGGINGFACEHUB_API_TOKEN=<your-huggingface-token>
   source scripts/setup_env.sh
   ```

2. Start the Docker containers with the previously built images:

   ```bash
   docker compose -f docker/compose.yaml up
   ```

3. Access the application:

   - Open your web browser and navigate to "http://<host-ip>:8101" to view the application dashboard.

## Verification

- Ensure the application is running by checking the Docker container status:

  ```bash
  docker ps
  ```

-  Access the application dashboard and verify that it is functioning as expected.

## Troubleshooting

- If you encounter any issues during the build or run process, check the Docker logs for errors:

  ```bash
  docker logs -f <container-id>
  ```
