#include <stdio.h>

// LAB TASK: Implement print_digit
void print_digit(int digit)
{
  if (digit < 10)
  {
    fputc('0' + digit, stdout);
  }
  else
  {
    fputc('a' + (digit - 10), stdout);
  }
}
// LAB TASK: Implement print_string
void print_string(char *s)
{
  while (*s != '\0')
  {
    fputc(*s, stdout);
    s++;
  }
}

int main(int argc, char *argv[])
{
  printf("print_digit test: \n"); // Not to use this in A1
  for (int i = 0; i <= 16; ++i)
  {
    print_digit(i);
    fputc(' ', stdout);
  }
  printf("\nprint_string test: \n"); // Not to use this in A1

  char *str = "Hello, 3410\n";
  print_string(str);
  return 0;
}