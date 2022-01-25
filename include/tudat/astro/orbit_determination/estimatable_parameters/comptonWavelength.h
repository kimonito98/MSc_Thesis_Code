//
// Created by Michael Plumaris on 21/01/2022.
//

#ifndef TUDATBUNDLE_COMPTONWAVELENGTH_H
#define TUDATBUNDLE_COMPTONWAVELENGTH_H

#include "tudat/astro/orbit_determination/estimatable_parameters/estimatableParameter.h"
#include "tudat/astro/relativity/metric.h"

// Michael
namespace tudat
{

    namespace estimatable_parameters
    {

//! Interface class for the estimation of a gravitational parameter
class ComptonWavelength: public EstimatableParameter< double >
{

public:

    //! Constructor
    ComptonWavelength(
            const std::shared_ptr< gravitation::GravityFieldModel > gravityFieldModel, const std::string& associatedBody ):
    EstimatableParameter< double >( compton_wavelength, associatedBody ),
    gravityFieldModel_( gravityFieldModel ){ }

    //! Destructor
    ~ComptonWavelength( ) { }

    //! Function to get the current value of the gravitational parameter that is to be estimated.
    /*!
     * Function to get the current value of the gravitational parameter that is to be estimated.
     * \return Current value of the gravitational parameter that is to be estimated.
     */
    double getParameterValue( )
    {
        return relativity::comptonWavelength;
    }

    //! Function to reset the value of the gravitational parameter that is to be estimated.
    /*!
     * Function to reset the value of the gravitational parameter that is to be estimated.
     * \param parameterValue New value of the gravitational parameter that is to be estimated.
     */
    void setParameterValue( double parameterValue )
    {
        relativity::comptonWavelength = 1 / std::sqrt( parameterValue );
    }

    //! Function to retrieve the size of the parameter (always 1).
    /*!
     *  Function to retrieve the size of the parameter (always 1).
     *  \return Size of parameter value (always 1).
     */
    int getParameterSize( )
    {
        return 1;
    }

protected:

private:
    std::shared_ptr< gravitation::GravityFieldModel > gravityFieldModel_;

};

} // namespace estimatable_parameters

} // namespace tudat

#endif //TUDATBUNDLE_COMPTONWAVELENGTH_H
