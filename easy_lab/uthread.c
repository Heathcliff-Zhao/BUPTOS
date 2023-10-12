#include "uthread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

static struct uthread *current_thread = NULL;
static struct uthread *main_thread = NULL;

/// @brief 切换上下文
/// @param from 当前上下文
/// @param to 要切换到的上下文
extern void thread_switch(struct context *from, struct context *to);

/// @brief 线程的入口函数
/// @param tcb 线程的控制块
/// @param thread_func 线程的执行函数
/// @param arg 线程的参数
void _uthread_entry(struct uthread *tcb, void (*thread_func)(void *),
                    void *arg);

/// @brief 清空上下文结构体
/// @param context 上下文结构体指针
static inline void make_dummpy_context(struct context *context) {
  memset((struct context *)context, 0, sizeof(struct context));
}


struct QueueNode {
  struct uthread *thread;
  struct QueueNode *next;
};

struct Queue {
  struct QueueNode *front;
  struct QueueNode *rear;
};

static struct Queue active_queue;

void init_queue(struct Queue *queue) {
  queue->front = queue->rear = NULL;
}

void enqueue(struct Queue *queue, struct uthread *thread) {
  struct QueueNode *newNode = (struct QueueNode *)malloc(sizeof(struct QueueNode));
  newNode->thread = thread;
  newNode->next = NULL;
  if (queue->rear == NULL) {
    queue->front = queue->rear = newNode;
  } else {
    queue->rear->next = newNode;
    queue->rear = newNode;
  }
}

struct uthread *dequeue(struct Queue *queue) {
  if (queue->front == NULL) {
    return NULL; // 队列为空
  }
  struct QueueNode *temp = queue->front;
  struct uthread *thread = temp->thread;
  queue->front = queue->front->next;
  if (queue->front == NULL) {
    queue->rear = NULL;
  }
  free(temp);
  return thread;
}

struct uthread *get_current_thread() {
  return current_thread;
}

void switch_to_next_thread(struct uthread *next_thread) {
  if (current_thread != NULL) {
    thread_switch(&(current_thread->context), &(next_thread->context));
  }
  
  current_thread = next_thread;
}
struct uthread *uthread_create(void (*func)(void *), void *arg, const char *thread_name) {
  struct uthread *uthread = NULL;
  int ret;

  ret = posix_memalign((void **)&uthread, 16, sizeof(struct uthread));
  if (0 != ret) {
    printf("Memory allocation error\n");
    exit(-1);
  }

  uthread->state = THREAD_INIT;
  uthread->name = thread_name;
  uthread->func = func;
  uthread->arg = arg;

  uintptr_t stack_top = (uintptr_t)(uthread->stack) + STACK_SIZE;
  stack_top = (stack_top & -16L) - 8;

  memset(&(uthread->context), 0, sizeof(struct context));
  uthread->context.rip = (long long)_uthread_entry;
  uthread->context.rsp = stack_top;
  uthread->context.rbp = stack_top;
  uthread->context.rbx = 0;
  uthread->context.r12 = 0;
  uthread->context.r13 = 0;
  uthread->context.r14 = 0;
  uthread->context.r15 = 0;

  uthread->context.rdi = (long long)uthread;
  uthread->context.rsi = (long long)func;
  uthread->context.rdx = (long long)arg;

  enqueue(&active_queue, uthread);
  return uthread;
}


void schedule() {
  if (active_queue.front == NULL) {
    return;
  }
  
  struct uthread *prev_thread = get_current_thread();
  struct uthread *next_thread = dequeue(&active_queue);

  assert(next_thread != NULL);
  next_thread->state = THREAD_RUNNING;

  current_thread = next_thread;

  if (prev_thread && prev_thread->state != THREAD_STOP && prev_thread != main_thread) {
    enqueue(&active_queue, prev_thread);
  }
  
  thread_switch(&(prev_thread->context), &(next_thread->context));
}


long long uthread_yield() {
  struct uthread *current_thread = get_current_thread();
  current_thread->state = THREAD_SUSPENDED;
  schedule();

  current_thread->state = THREAD_RUNNING;
  
  return 0;
}


void uthread_resume(struct uthread *tcb) {
  struct uthread *current_thread = get_current_thread();
  if (tcb == NULL || tcb == current_thread) {
    return;
  }
  if (current_thread != NULL && current_thread->state != THREAD_STOP) {
    struct QueueNode *temp = active_queue.front;
    while (temp != NULL) {
      temp = temp->next;
    }
    thread_switch(&(current_thread->context), &(tcb->context));
  }

}


void thread_destory(struct uthread *tcb) {
  free(tcb);
}

void _uthread_entry(struct uthread *tcb, void (*thread_func)(void *), void *arg) {

  tcb->state = THREAD_RUNNING;

  thread_func(arg);
  tcb->state = THREAD_STOP;

  thread_destory(tcb);

  if (active_queue.front == NULL) {
    thread_switch(&(tcb->context), &(main_thread->context));
  } else {
    schedule();
  }
}

void init_uthreads() {
  main_thread = malloc(sizeof(struct uthread));
  main_thread->state = THREAD_RUNNING;
  main_thread->name = "main";
  make_dummpy_context(&main_thread->context);
  current_thread = main_thread;
  init_queue(&active_queue);
}