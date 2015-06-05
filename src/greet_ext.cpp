#include "greet.h";
#include <boost/python.hpp>;
 
BOOST_PYTHON_MODULE(greet_ext)
{
    using namespace boost::python;
    def( "greet", greet );
}
