//
// Created by Michael Plumaris on 21/01/2022.
//

#include "tudat/astro/orbit_determination/acceleration_partials/comptonWavelengthAccelerationPartial.h"


namespace tudat
{

namespace acceleration_partials
{

//! Calculates partial derivative of point mass gravitational acceleration wrt the position of body undergoing acceleration.
Eigen::Matrix3d calculatePartialOfCentralGravityComptonWrtPositionOfAcceleratedBody(
        const Eigen::Vector3d& acceleratedBodyPosition,
        const Eigen::Vector3d& acceleratingBodyPositions,
        double gravitationalParameter,
        double lambda )
{
    // Calculate relative position
    Eigen::Vector3d relativePosition = acceleratedBodyPosition - acceleratingBodyPositions;

    // Calculate partial (Montenbruck & Gill, Eq. 7.56)
    double relativePositionNorm = relativePosition.norm( );
    double invSquareOfPositionNorm = 1.0 / ( relativePositionNorm * relativePositionNorm );
    double invCubeOfPositionNorm = invSquareOfPositionNorm / relativePositionNorm;
    Eigen::Matrix3d partialMatrix = -gravitationalParameter *
                                    ( Eigen::Matrix3d::Identity( ) * invCubeOfPositionNorm -
                                      ( 3.0 * invSquareOfPositionNorm * invCubeOfPositionNorm ) * relativePosition * relativePosition.transpose( ) ) +
                                      // additional part
                                      gravitationalParameter / (2 * lambda * lambda ) *
                                      ( Eigen::Matrix3d::Identity( ) / relativePositionNorm - relativePosition * invSquareOfPositionNorm ) ;

    return partialMatrix;
}

//! CAREFUL: wrt 1 / lambda**2 !!!
Eigen::Vector3d computePartialOfCentralGravityComptonWrtComptonWavelength( const Eigen::Vector3d& acceleratedBodyPosition,
                                                                         const Eigen::Vector3d& acceleratingBodyPosition,
                                                                         const double gravitationalParameter )
{
    // Calculate relative position
    Eigen::Vector3d relativePosition = acceleratingBodyPosition - acceleratedBodyPosition;

    // Calculate partial (Montenbruck & Gill, Eq. 7.76)
    double positionNorm = relativePosition.norm( );
    Eigen::Vector3d partialMatrix = gravitationalParameter * relativePosition / ( 2 * positionNorm );
    return partialMatrix;
}


//! Constructor
comptonWavelengthAccelerationPartial::comptonWavelengthAccelerationPartial(
        const std::shared_ptr< gravitation::CentralGravitationalAccelerationModel3d > gravitationalAcceleration,
        const std::string acceleratedBody,
        const std::string acceleratingBody ):
        AccelerationPartial( acceleratedBody, acceleratingBody, basic_astrodynamics::central_gravity_Compton )
{
    accelerationUpdateFunction_ =
            std::bind( &basic_astrodynamics::AccelerationModel< Eigen::Vector3d>::updateMembers, gravitationalAcceleration, std::placeholders::_1 );

    gravitationalParameterFunction_ = gravitationalAcceleration->getGravitationalParameterFunction( );
    comptonWavelengthFunction_ = gravitationalAcceleration->getGravitationalParameterFunction( );
    centralBodyState_ = gravitationalAcceleration->getStateFunctionOfBodyExertingAcceleration( );
    acceleratedBodyState_ = gravitationalAcceleration->getStateFunctionOfBodyUndergoingAcceleration( );
    accelerationUsesMutualAttraction_ = gravitationalAcceleration->getIsMutualAttractionUsed( );
}

//! Function for setting up and retrieving a function returning a partial w.r.t. a double parameter.
std::pair< std::function< void( Eigen::MatrixXd& ) >, int >
comptonWavelengthAccelerationPartial::getParameterPartialFunction(
        std::shared_ptr< estimatable_parameters::EstimatableParameter< double > > parameter )

{
    std::pair< std::function< void( Eigen::MatrixXd& ) >, int > partialFunctionPair;

    // Check dependencies.
    if( parameter->getParameterName( ).first ==  estimatable_parameters::compton_wavelength )
    {
        // If parameter is gravitational parameter, check and create dependency function .
        partialFunctionPair = getComptonWavelengthPartialFunction( parameter->getParameterName( ) );
    }
    else
    {
        partialFunctionPair = std::make_pair( std::function< void( Eigen::MatrixXd& ) >( ), 0 );
    }

    return partialFunctionPair;
}

//! Function to create a function returning the current partial w.r.t. a gravitational parameter.
std::pair< std::function< void( Eigen::MatrixXd& ) >, int >
comptonWavelengthAccelerationPartial::getComptonWavelengthPartialFunction(
        const estimatable_parameters::EstimatebleParameterIdentifier& parameterId )
{
    std::function< void( Eigen::MatrixXd& ) > partialFunction;
    int numberOfColumns = 0;

    // Check if parameter is gravitational parameter.
    if( parameterId.first ==  estimatable_parameters::compton_wavelength )
    {
        // Check if parameter body is central body.
        if( parameterId.second.first == acceleratingBody_ )
        {
            partialFunction = std::bind( &comptonWavelengthAccelerationPartial::wrtComptonWavelengthOfCentralBody,
                                         this, std::placeholders::_1 );
            numberOfColumns = 1;

        }
        // Check if parameter body is accelerated body, and if the mutual acceleration is used.
        else
        {
            throw std::runtime_error( "Error when making Compton wavelength partial, associated body doesn't have potential" );
        }
    }

    return std::make_pair( partialFunction, numberOfColumns );
}

//! Function to calculate central gravity partial w.r.t. central body gravitational parameter
void comptonWavelengthAccelerationPartial::wrtComptonWavelengthOfCentralBody( Eigen::MatrixXd& comptonWavelengthPartial )
{
    comptonWavelengthPartial = computePartialOfCentralGravityComptonWrtComptonWavelength(
            currentAcceleratedBodyState_, currentCentralBodyState_, currentGravitationalParameter_);
}



}

}
