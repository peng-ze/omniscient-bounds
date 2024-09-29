package main

import "fmt"
import "os"
import "strings"
import "strconv"
import (
    "net"
	"io"
	"os/exec"
	"time"
	"encoding/json"
	"context"
	"os/signal"
	"syscall"
)

type CommandData struct {
    Command string            `json:"command"`
    Env     map[string]string `json:"env"`
}


func socketListener(socketPath string, ch chan CommandData) {
    if _, err := os.Stat(socketPath); err == nil {
        fmt.Fprintf(os.Stderr, "Error: socket file %s already exists. Exiting...\n", socketPath)
        os.Exit(1) 
    }

    listener, err := net.Listen("unix", socketPath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error creating listener: %v\n", err)
        os.Exit(1)
    }
    defer listener.Close()

    fmt.Println("Server is listening on", socketPath)

    for {
        conn, err := listener.Accept()
        if err != nil {
            fmt.Println("Error accepting connection:", err)
            continue
        }

        go handleConnection(conn, ch)
    }
}

func handleConnection(conn net.Conn, ch chan CommandData) {
    defer conn.Close()

    var data CommandData
    decoder := json.NewDecoder(conn)
    if err := decoder.Decode(&data); err != nil {
        fmt.Println("Error decoding data:", err)
        return
    }

	ch <- data
}

func createLogger(path string, gpu_id string, local_id int) (*os.File, string, error) {
    logFileName := fmt.Sprintf("%s/gpu_%s_worker_%d.log", path, gpu_id, local_id)

    logFile, err := os.OpenFile(logFileName, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
    if err != nil {
        return nil, "", fmt.Errorf("error creating log file: %v", err)
    }

    return logFile, logFileName, nil
}

func setEnv(env []string, key, value string) []string {
    for i, v := range env {
        if strings.HasPrefix(v, key+"=") {
            env[i] = fmt.Sprintf("%s=%s", key, value)
            return env
        }
    }
    return append(env, fmt.Sprintf("%s=%s", key, value))
}

func execute(ctx context.Context, gpu_id string, logFile *os.File, command string, env map[string]string){
	currentDateTime := time.Now().Format("2006-01-02_15-04-05")
	commandSummary := fmt.Sprintf("CUDA_VISIBLE_DEVICES=%s %s", gpu_id, command)
	logFile.WriteString(currentDateTime + "\n")
	logFile.WriteString(commandSummary + "\n")


	envVars := make([]string, 0)
	for key, value := range env {
        envVars = setEnv(envVars, key, value)
    }
	envVars = setEnv(envVars, "CUDA_VISIBLE_DEVICES", gpu_id)

	logFile.WriteString(fmt.Sprint(envVars) + "\n\n")

	parts := strings.Fields(command)
    cmdName := parts[0]
    cmdArgs := parts[1:]

	cmd := exec.CommandContext(ctx, cmdName, cmdArgs...)
    cmd.Env = envVars

	cmd.Stdout = io.MultiWriter(logFile, os.Stdout)
    cmd.Stderr = io.MultiWriter(logFile, os.Stderr) 

	if err := cmd.Run(); err != nil {
        logFile.WriteString(fmt.Sprintf("Error executing command: %v\n", err))
    }
}

func worker(ctx context.Context, log_path string, gpu_id string, local_id int, channel chan CommandData, feedback_channel chan string) {
	logFile, logFilePath, _ := createLogger(log_path, gpu_id, local_id)
	fmt.Println(logFilePath)
	defer logFile.Close()

	for command := range channel {
		feedback_channel <- "Worker " + strconv.Itoa(local_id) + " at GPU " + gpu_id + " takes care of command: " + command.Command
		execute(ctx, gpu_id, logFile, command.Command, command.Env)
	}
}

func main() {
	gpu_ids := strings.Split(os.Getenv("CUDA_VISIBLE_DEVICES"), ",")
	workers_per_gpu, _ := strconv.Atoi(os.Getenv("WORKERS_PER_GPU"))
	log_root := os.Getenv("LOG_ROOT")
	currentDateTime := time.Now().Format("2006-01-02_15-04-05")
	log_path := fmt.Sprintf("%s/logs_%s_%s", log_root, os.Getenv("IDENTIFIER"), currentDateTime)
	os.MkdirAll(log_path, 0755)

	channel := make(chan CommandData)
	feedback_channel := make(chan string)

	ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    sigs := make(chan os.Signal, 1)
    signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
    go func() {
        <-sigs
        feedback_channel <- "Received termination signal, shutting down..."

        cancel() 
    }()


	for _, gpu_id := range gpu_ids {
		for i := 0; i < workers_per_gpu; i++ {
			go worker(ctx, log_path, gpu_id, i, channel, feedback_channel)
		}
	}

	socketPath := os.Getenv("SOCKET_PATH")
	go socketListener(socketPath, channel)
	defer os.Remove(socketPath)


	for feedback := range feedback_channel {
		fmt.Println(feedback)
		if feedback == "Received termination signal, shutting down..." {
			break
		}
	}
}