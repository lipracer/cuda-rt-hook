#define EXPORT __attribute__((__visibility__("default")))
EXPORT void* mmalloc(int);

int main(int argc, char** argv) {
    mmalloc(0);
    return 0;
}