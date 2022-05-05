#include  <stdio.h>
#include  <stdlib.h>

void read_from_file(void *ptr, size_t size, size_t n, char *filename, int check){
  
	FILE *fp;
	int num;
  	
	if (NULL != (fp = fopen(filename, "r"))){
		
		num = fread(ptr, size, n, fp);
		
		if (num != n){
			
			if ( ferror(fp) ) {
				printf("Erron reading %s\n", filename); 
				exit(0);
			} else if ( feof(fp) ) {
				printf("EOF found in reading %s\n", filename);exit(0);}
		}	
  	
	} else {
		
		printf("Error opening %s\n", filename); exit(0);
		
	}
	
	if (1 == check) {

		fgetc(fp);
		if (!feof(fp)) {
			printf("Elements left in %s file\n", filename); 
			exit(0);

		}
	}

	fclose(fp);

}

void write_to_file(void *ptr, size_t size, size_t n, char *filename){

	FILE *fp;
	int num;
  	
	if (NULL != (fp = fopen(filename, "w"))){
		
		num = fwrite(ptr, size, n, fp);
		if (num != n){
			
			printf("Erron writing to %s\n", filename); exit(0);
			
		}	
  	
	} else {
		
		printf("Error opening %s\n", filename); exit(0);
		
	}

	fclose(fp);

}


