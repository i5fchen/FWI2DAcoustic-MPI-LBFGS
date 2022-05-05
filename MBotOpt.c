#include <mpi.h>
#include "header.h"
#include "Mpar.h"
#include "lbfgs.h"
#include "fwi_files_path.h"
#include "read_write.h"
int main(int argc, char **argv)
{
    int i, j, k, end, bound, count=0, ret=0, n=NX*NZ;
    float fx, step, width, dg;
    const float dec = 0.5, inc = 2.1; 
	float finit=0, dginit=0, dgtest=0; 
    lbfgs_parameter_t param;
    float *x = NULL, *xp = NULL, *g = NULL,  *d = NULL, *gp=NULL;
    iteration_data_t *lm = NULL, *it = NULL;
    float ys, yy;
    float xnorm, gnorm, beta, *step_arr = NULL;
    int my_id, root_process, ierr, num_procs, an_id;
    int signal_exit1, signal_exit2, signal_exit3, signal_break;
   	
	char filename[500];
	char *HEAD=argv[1]; 
    MPI_Status status;
    root_process = 0;
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (my_id==root_process){
        memcpy(&param, &_defparam, sizeof(param));
        xp = (float*)calloc(n, sizeof(float));
        gp = (float*)calloc(n, sizeof(float));
        d = (float*)calloc(n, sizeof(float));
        step_arr =(float *)calloc(param.max_iterations+1,sizeof(float));

    }
    /* Allocate limited memory storage. */
    if (my_id==root_process){
        lm = (iteration_data_t*)calloc(param.m, sizeof(iteration_data_t));
        /* Initialize the limited memory. */
        for (i = 0;i < param.m;++i) {
            it = &lm[i];
            it->alpha = 0;
            it->ys = 0;
            it->s = (float*)calloc(n, sizeof(float));
            it->y = (float*)calloc(n, sizeof(float));
        }
    }
    
	fwi_plan *plan = (fwi_plan *)malloc(sizeof(fwi_plan)); 
	fwi_init(plan, my_id); 
	x = &plan->model->md[0][0];
	g = plan->grad->body; 

    grad4_1shot(plan,k);

	if (0 == plan->grad->id){ 
		
		write_to_file(plan->grad->body, sizeof(float), n, "g_prec_0.bin");	
		   
    	printf("The initial msf is %f (l2) and %f\n", plan->msf->val[0], plan->msf->val[1]); 
	    fflush(stdout);
	}
    
	if (root_process==my_id){
        
        vecncpy(d, g, n);
        vec2norm(&xnorm, x, n);
        vec2norm(&gnorm, g, n);

        step = 0.8;//6.25;
        k = 1;
        end = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (;;) {
        if (root_process==my_id){
            /* Store the current position and gradient vectors. */
            veccpy(xp, x, n);
            veccpy(gp, g, n);

            /* Search for an optimal step. */
            count = 0;
            /* Compute the initial gradient in the search direction. */
            vecdot(&dginit, g, d, n);

            /* The initial value of the objective function. */
            finit = plan->msf->val[1];
            dgtest = param.ftol * dginit;
        }
        for (;;){ //line search
            if (my_id==root_process){
                veccpy(x, xp, n);
                vecadd(x, d, step, n);
                               
            }
            
			MPI_Bcast(x, n, MPI_FLOAT, root_process, MPI_COMM_WORLD);
            grad4_1shot(plan,k);
			
            if (root_process==my_id){
               fx = plan->msf->val[1];
                ++count;
                if (fx > finit + step * dgtest){


                    signal_break = 3; // do not satisfied Armijo
                    for (an_id=1; an_id<num_procs; an_id++)
                        MPI_Send( &signal_break, 1, MPI_INT, an_id, send_data_tag0, MPI_COMM_WORLD);
                    width = dec;
                   
                }else{
                   
                    /* Check the Wolfe condition. */
                    vecdot(&dg, g, d, n);
                    if (dg < param.wolfe * dginit){

                        signal_break = 2; // do not satisfied the curvature condition.
                        for (an_id=1; an_id<num_procs; an_id++)
                            MPI_Send(&signal_break, 1, MPI_INT, an_id, send_data_tag0, MPI_COMM_WORLD);
                        width = inc;
                    }else{
                        /* Check the strong Wolfe condition. */
                        if(dg > -param.wolfe * dginit) {

                            signal_break = 1; // do not satisfied the strong curvature condition
                            for (an_id=1; an_id<num_procs; an_id++)
                                MPI_Send( &signal_break, 1, MPI_INT, an_id, send_data_tag0, MPI_COMM_WORLD);
                            width = dec;
                        }else{
                            
                            /* Exit with the strong Wolfe condition. */
                            signal_break = 0;
                            for (an_id=1; an_id<num_procs; an_id++)
                                MPI_Send( &signal_break, 1, MPI_INT, an_id, send_data_tag0, MPI_COMM_WORLD);
                            break;
                        }
                    }
                }
                if (param.max_linesearch <= count) {
                    /* Maximum number of iteration. */
                    ret=LBFGSERR_MAXIMUMLINESEARCH; signal_exit1 = 0;
                    for (an_id=1; an_id<num_procs; an_id++)
                        MPI_Send( &signal_exit1, 1, MPI_INT, an_id, send_data_tag1, MPI_COMM_WORLD);
                    goto lbfgs_exit;
                }else{
                    signal_exit1 = -1024;
                    for (an_id=1; an_id<num_procs; an_id++)
                        MPI_Send( &signal_exit1, 1, MPI_INT, an_id, send_data_tag1, MPI_COMM_WORLD);
                }
                step *= width;
            }else{
                ierr = MPI_Recv(&signal_break, 1, MPI_INT, root_process, send_data_tag0, MPI_COMM_WORLD, &status);
                if (signal_break == 0) break;
                ierr = MPI_Recv(&signal_exit1, 1, MPI_INT, root_process, send_data_tag1, MPI_COMM_WORLD, &status);
                if (signal_exit1 == 0) goto lbfgs_exit;
            }
            //if (my_id!=root_process && signal_exit == 0) goto lbfgs_exit;
        }
        if (my_id==root_process){
            //ptr_fx[k] = fx; ptr_fx0[k] = fx0;
            vec2norm(&xnorm, x, n); vec2norm(&gnorm, g, n);
   
            step_arr[k] = step;
        	
            if (gnorm / xnorm <= param.epsilon) {
                /* Convergence. */
                ret = LBFGS_SUCCESS; signal_exit2 = 0;
                for (an_id=1; an_id<num_procs; an_id++)
                    MPI_Send( &signal_exit2, 1, MPI_INT, an_id, send_data_tag2, MPI_COMM_WORLD);
                goto lbfgs_exit;
            }else{
                signal_exit2 = -1024;
                for (an_id=1; an_id<num_procs; an_id++)
                   MPI_Send( &signal_exit2, 1, MPI_INT, an_id, send_data_tag2, MPI_COMM_WORLD);
            }
            if (param.max_iterations != 0 && param.max_iterations < k+1){
                /* Maximum number of iterations. */
                ret = LBFGSERR_MAXIMUMITERATION; signal_exit3 = 0;
                for (an_id=1; an_id<num_procs; an_id++)
                    MPI_Send( &signal_exit3, 1, MPI_INT, an_id, send_data_tag3, MPI_COMM_WORLD);
                goto lbfgs_exit;
            }else{
                signal_exit3 = -1024;
                for (an_id=1; an_id<num_procs; an_id++)
                    MPI_Send(&signal_exit3, 1, MPI_INT, an_id, send_data_tag3, MPI_COMM_WORLD);
            }
            it = &lm[end];
            vecdiff(it->s, x, xp, n);
            vecdiff(it->y, g, gp, n);
            vecdot(&ys, it->y, it->s, n);
            vecdot(&yy, it->y, it->y, n);
            it->ys = ys;
            bound = (param.m <= k) ? param.m : k;
            ++k;
            end = (end + 1) % param.m;
            vecncpy(d, g, n);
            j = end;
            for (i = 0;i < bound;++i) {
                j = (j + param.m - 1) % param.m;
                it = &lm[j];
                vecdot(&it->alpha, it->s, d, n);
                it->alpha /= it->ys;
                vecadd(d, it->y, -it->alpha, n);
            }
            vecscale(d, ys / yy, n);
            for (i = 0;i < bound;++i) {
                it = &lm[j];
                vecdot(&beta, it->y, d, n);
                beta /= it->ys;
                vecadd(d, it->s, it->alpha - beta, n);
                j = (j + 1) % param.m;
            }
            if (k<4) step = 0.2; else step = 0.5*(step_arr[k-1]+step_arr[k-2]);

        }else{
            ierr = MPI_Recv(&signal_exit2, 1, MPI_INT, root_process, send_data_tag2, MPI_COMM_WORLD, &status);
            ierr = MPI_Recv(&signal_exit3, 1, MPI_INT, root_process, send_data_tag3, MPI_COMM_WORLD, &status);
            if (signal_exit3*signal_exit2 == 0) goto lbfgs_exit;
        }
    }

    lbfgs_exit:
    if (my_id==root_process){
        /* Free memory blocks used by this function. */
        if (lm != NULL) {
            for (i = 0;i < param.m;++i) {
                free(lm[i].s);
                free(lm[i].y);
            }
            free(lm);
        }
        free(d);
        free(gp);
        free(g);
        free(xp);
    }
    free(x);
    ierr = MPI_Finalize();
    
    return 0;
}
