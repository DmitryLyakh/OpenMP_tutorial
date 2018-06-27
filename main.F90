module linear_algebra
 use omp_lib
 implicit none
 public

contains

 subroutine v_set(n,v,val)
  integer, intent(in):: n       !vector dimension
  real(8), intent(inout):: v(n) !vector
  real(8), intent(in):: val     !value to set for each element of the vector
  integer:: i

!$OMP PARALLEL DEFAULT(NONE) SHARED(n,v,val) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
  do i=1,n
   v(i)=val
  enddo
!$OMP END DO
!$OMP END PARALLEL
 end subroutine v_set

 subroutine m_set(n,m,w,val)
  integer, intent(in):: n,m       !matrix dimensions
  real(8), intent(inout):: w(n,m) !matrix
  real(8), intent(in):: val       !value to set for each element of the matrix
  integer:: i,j

!$OMP PARALLEL DEFAULT(NONE) SHARED(n,m,w,val) PRIVATE(i,j)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
  do j=1,m
   do i=1,n
    w(i,j)=val
   enddo
  enddo
!$OMP END DO
!$OMP END PARALLEL
 end subroutine m_set

 subroutine v_norm2(n,v,d)
  integer, intent(in):: n    !vector dimension
  real(8), intent(in):: v(n) !vector
  real(8), intent(out):: d   !vector norm
  integer:: i

  d=0d0
!$OMP PARALLEL DEFAULT(NONE) SHARED(n,v) PRIVATE(i) REDUCTION(+:d)
!$OMP DO SCHEDULE(GUIDED)
  do i=1,n
   d=d+v(i)*v(i)
  enddo
!$OMP END DO
!$OMP END PARALLEL
  d=sqrt(d)
 end subroutine v_norm2

 subroutine m_norm2(n,m,w,d)
  integer, intent(in):: n,m    !matrix dimensions
  real(8), intent(in):: w(n,m) !matrix
  real(8), intent(out):: d     !matrix norm
  integer:: i,j

  d=0d0
!$OMP PARALLEL DEFAULT(NONE) SHARED(n,m,w) PRIVATE(i,j) REDUCTION(+:d)
!$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
  do j=1,m
   do i=1,n
    d=d+w(i,j)*w(i,j)
   enddo
  enddo
!$OMP END DO
!$OMP END PARALLEL
  d=sqrt(d)
 end subroutine m_norm2

 subroutine vv_mul(n,v1,v2,d)
  integer, intent(in):: n     !vector dimension
  real(8), intent(in):: v1(n) !vector 1
  real(8), intent(in):: v2(n) !vector 2
  real(8), intent(out):: d    !dot-product
  integer:: i,nth

  d=0d0
!$OMP PARALLEL DEFAULT(NONE) SHARED(nth,n,v1,v2) PRIVATE(i) REDUCTION(+:d)
!$OMP MASTER
  nth=omp_get_num_threads()
!$OMP END MASTER
!$OMP DO SCHEDULE(GUIDED)
  do i=1,n
   d=d+v1(i)*v2(i)
  enddo
!$OMP END DO
!$OMP END PARALLEL
  print *,'vv_mul: num_threads = ',nth
 end subroutine vv_mul

 subroutine mv_mul(variant,n,m,w1,v1,v2)
  integer, intent(in):: variant   !variant of the code: {1,2}
  integer, intent(in):: n,m       !matrix dimensions
  real(8), intent(in):: w1(n,m)   !input matrix
  real(8), intent(in):: v1(m)     !input vector
  real(8), intent(inout):: v2(n)  !output vector
  integer:: i,j
  real(8):: tmp

  select case(variant)
  case(1)
!$OMP PARALLEL DEFAULT(NONE) SHARED(n,m,w1,v1,v2) PRIVATE(i,j,tmp)
!$OMP DO SCHEDULE(GUIDED)
   do i=1,n
    tmp=0d0
    do j=1,m
     tmp=tmp+w1(i,j)*v1(j)
    enddo
    v2(i)=tmp
   enddo
!$OMP END DO
!$OMP END PARALLEL
  case(2)
!$OMP PARALLEL DEFAULT(NONE) SHARED(n,m,w1,v1,v2) PRIVATE(i,j,tmp)
!$OMP DO SCHEDULE(GUIDED)
   do i=1,n
    v2(i)=0d0
   enddo
!$OMP END DO
   do j=1,m
    tmp=v1(j)
!$OMP DO SCHEDULE(GUIDED)
    do i=1,n
     v2(i)=v2(i)+w1(i,j)*tmp
    enddo
!$OMP END DO
   enddo
!$OMP END PARALLEL
  end select
 end subroutine mv_mul

 subroutine mm_mul(variant,n,m,l,w1,w2,w3)
  integer, intent(in):: variant         !variant of the code: {1}
  integer, intent(in):: n,m,l           !matrix multiplication dimensions
  real(8), intent(in):: w1(n,l),w2(l,m) !input matrices
  real(8), intent(inout):: w3(n,m)      !output matrix
  integer:: i,j,k
  real(8):: tmp

  select case(variant)
  case(1)
!$OMP PARALLEL DEFAULT(NONE) SHARED(n,m,l,w1,w2,w3) PRIVATE(i,j,k,tmp)
!$OMP DO SCHEDULE(GUIDED) COLLAPSE(2)
   do i=1,n
    do j=1,m
     tmp=0d0
     do k=1,l
      tmp=tmp+w1(i,k)*w2(k,j)
     enddo
     w3(i,j)=tmp
    enddo
   enddo
!$OMP END DO
!$OMP END PARALLEL
  end select
 end subroutine mm_mul

 function mm_flops(n,m,l,tm) result(gflops)
  real(8):: gflops            !GFlop/s
  integer, intent(in):: n,m,l !matrix multiplication dimensions
  real(8), intent(in):: tm    !time taken by matrix multiplications

  gflops=2d0*real(n,8)*real(m,8)*real(l,8)/tm/1d9
 end function mm_flops

end module linear_algebra

program main
 use omp_lib
 use linear_algebra
 integer, parameter:: n=1000000,m=100,l=200
 real(8), allocatable:: v1(:),v2(:),v3(:),w1(:,:),w2(:,:),w3(:,:)
 real(8):: d,tms,tmf

!Allocate data:
 allocate(w3(n,m),w1(n,l),w2(l,m),v3(n),v2(n),v1(m))

!Set the desired number of threads:
 call omp_set_num_threads(1)

!Set vector 1:
 tms=omp_get_wtime()
 call v_set(m,v1,1d0)
 tmf=omp_get_wtime()
 print *,'Vector 1 set; Time (s) = ',tmf-tms

!Set vector 2:
 tms=omp_get_wtime()
 call v_set(n,v2,0d0)
 tmf=omp_get_wtime()
 print *,'Vector 2 set; Time (s) = ',tmf-tms

!Set vector 3:
 tms=omp_get_wtime()
 call v_set(n,v3,3d0)
 tmf=omp_get_wtime()
 print *,'Vector 3 set; Time (s) = ',tmf-tms

!Set matrix 1:
 tms=omp_get_wtime()
 call m_set(n,l,w1,1d0)
 tmf=omp_get_wtime()
 print *,'Matrix 1 set; Time (s) = ',tmf-tms

!Set matrix 2:
 tms=omp_get_wtime()
 call m_set(l,m,w2,1d0)
 tmf=omp_get_wtime()
 print *,'Matrix 2 set; Time (s) = ',tmf-tms

!Set matrix 3:
 tms=omp_get_wtime()
 call m_set(n,m,w3,1d0)
 tmf=omp_get_wtime()
 print *,'Matrix 3 set; Time (s) = ',tmf-tms

!Matrix-vector multiplication:
 tms=omp_get_wtime()
 call mv_mul(1,n,m,w3,v1,v2)
 tmf=omp_get_wtime()
 call v_norm2(n,v2,d)
 print *,'Matrix-vector product norm = ',d,'; Time (s) = ',tmf-tms

!Vector-Vector multiplication (dot-product):
 tms=omp_get_wtime()
 call vv_mul(n,v2,v3,d)
 tmf=omp_get_wtime()
 print *,'Vector-vector product = ',d,'; Time (s) = ',tmf-tms

!Matrix-matrix multiplication:
 call m_set(n,m,w3,0d0) !reset matrix 3 to zero
 tms=omp_get_wtime()
 call mm_mul(1,n,m,l,w1,w2,w3)
 tmf=omp_get_wtime()
 call m_norm2(n,m,w3,d)
 print *,'Matrix-matrix product norm = ',d,'; Time (s) = ',tmf-tms,'; GFlop/s = ',mm_flops(n,m,l,tmf-tms)

!Deallocate data:
 deallocate(w3,w1,w2,v2,v1)

end program main
